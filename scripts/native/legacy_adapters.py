"""Nine explicit adapters: native typed requests where supported; lossless mapping otherwise."""
import asyncio
import json
import os
import sys
from pathlib import Path
from common import PACKAGE, digest, sha
from protocol import text, portable_view, options, labels

SOURCE = PACKAGE/'vendor'
CAP = 32768

class InputTooLong(ValueError): pass

def guard_count(n):
    if n > CAP: raise InputTooLong(f'Encoded input {n} exceeds {CAP}; no truncation')

class Adapter:
    def __init__(self, name, model_path, base_model=None):
        import torch
        self.torch = torch; self.name = name; self.last_render = None
        torch.set_num_threads(4); torch.manual_seed(1234)
        sys.path.insert(0,str(SOURCE/name))
        if name == 'openjev': self.load_openjev(model_path)
        elif name == 'kev':
            from kev.checkpoint import Checkpoint, LoadOptions
            if base_model is None: raise ValueError('Kev requires --base-model')
            ck=Checkpoint(str(model_path)); ck.meta.base=str(base_model); ck.meta.base_revision=None
            tok,model=ck.load('cuda',LoadOptions(dtype=torch.bfloat16))
            self.engine=(tok,model,ck)
        elif name=='semif':
            from semif_phase1.core import load_causal_model
            self.engine=load_causal_model(str(model_path),'local-config-sha256:'+sha(Path(model_path)/'config.json'),device='cuda',dtype='bfloat16')
        elif name=='anyjev':
            from anyjev import Decider,Question
            from anyjev.backends.hf import HFBackend
            self.engine=(Decider(HFBackend(str(model_path),device='cuda',dtype='bfloat16',batch_size=1),level='raw'),Question)
            self.guard_anyjev()
        elif name=='gliner':
            from gliner2 import AutoExtractor
            self.engine=AutoExtractor.from_pretrained(str(model_path)).to('cuda').eval()
            def hook(module,positional,keywords):
                x=keywords.get('input_ids',positional[0] if positional else None)
                if x is not None: guard_count(x.shape[-1])
            self.engine.encoder.register_forward_pre_hook(hook,with_kwargs=True)
        else: raise ValueError(name)

    def guard_anyjev(self):
        backend=self.engine[0].backend
        original=backend.next_token_logprobs
        shared=backend.score_shared
        def capture(prompts):
            for p in prompts:
                n=len(backend.tokenizer.encode(p,add_special_tokens=False));guard_count(n)
                self.last_render.append({'prompt':p,'tokens':n})
        def guarded(prompts,*a,**kw):
            capture(prompts);return original(prompts,*a,**kw)
        def guarded_shared(groups,*a,**kw):
            capture([prefix+suffix for prefix,suffixes in groups for suffix in suffixes])
            return shared(groups,*a,**kw)
        backend.next_token_logprobs=guarded;backend.score_shared=guarded_shared

    def load_openjev(self, path):
        from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion
        from openjev.config import Settings
        from openjev.engine import Engine
        path=str(path)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.engine=Engine(Settings(tokenizer=path,canvas=64,max_images=0,warmup=False),self.tokenizer)
        self.model, info=DiffusionGemmaForBlockDiffusion.from_pretrained(path,dtype=self.torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='eager',experts_implementation='eager',
            local_files_only=True,output_loading_info=True)
        for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']: assert not info.get(k),(k,info[k])
        self.model.eval()
        assert all(p.device.type=='cuda' for p in self.model.parameters())

    def predict(self, body):
        q=body['questions']['decision']; kind=q['type']; t=self.torch
        if self.name=='kev':
            from kev.api import SystemOneRequest,to_record,to_answers
            tok,model,ck=self.engine
            req=SystemOneRequest(state=body['state'],questions=body['questions'],model='kev-latest')
            record,meta=to_record(req)
            encoded=model.encode(tok,record,max_state=CAP,max_branch=CAP,strict=True)
            self.last_render={'record':record,'token_ids':encoded['ids']}
            with t.inference_mode(): ps=model.probs(encoded)
            raw=ps[0].float().cpu().tolist()
            native=to_answers([raw],meta)['decision']
            p=dict(zip(meta[0]['keys'],raw)) if kind=='choice' else {'A':raw[1],'B':raw[0]}
            return p,{'native_answer':native,'head_temperature':ck.meta.temperature,
                'input_tokens':len(encoded['ids']),'adapter':'native_choice_noul','probabilities':'unrounded native head probabilities'}
        if self.name=='anyjev':
            decider, Question=self.engine; v=portable_view(body)
            if kind=='noul':
                question=Question.noul(v['question']+'\nAnswer Yes for criteria.true; No for criteria.false.',name='decision')
            else:
                question=Question.choice(v['question'],[f"{o['id']}: {o['description']}" for o in v['options']],name='decision')
            self.last_render=[]
            d=decider.decide(v['state'],[question])['decision']
            assert self.last_render,'No actual backend prompt captured'
            p=dict(zip(labels(body),map(float,d.probs)))
            return p,{'level':d.level,'native_type':question.kind,'adapter':'full_question_json_with_native_noul_yes_no',
                'input_tokens':max(x['tokens'] for x in self.last_render)}
        if self.name=='semif':
            from semif_phase1.direct import score
            from semif_phase1.core import direct_messages
            model,tok,meta=self.engine;v=portable_view(body);v['id']='decision'
            self.last_render=direct_messages(v)
            r=score(model,tok,v,meta,max_tokens=CAP)
            return dict(zip(r['option_ids'],r['probabilities'])),{**r,'adapter':'full_question_json_categorical_primitive_not_native_noul'}
        if self.name=='gliner': return self.predict_gliner(body)
        if self.name=='openjev': return self.predict_openjev(body)
        raise ValueError(self.name)

    def predict_gliner(self,body):
        from gliner2.training.trainer import ExtractorCollator
        t=self.torch;e=self.engine
        # Entire original state/questions JSON is the text input, with full criterion descriptions in the classifier.
        content=json.dumps({k:body[k] for k in ['state','questions']},ensure_ascii=False,allow_nan=False)
        opts=options(body);names=[f'{k}: {text(v)}' for k,v in opts]
        assert len(names)==len(set(names))
        schema=e._classification_schema({'decision':names})
        schemas,_=e._build_schema_dicts_and_metadata([schema])
        collator=ExtractorCollator(e.processor,is_training=False,architecture=e.architecture)
        batch=collator([(content,schemas[0])]);guard_count(batch.input_ids.shape[-1])
        self.last_render={'text':content,'schema':schemas[0],'token_ids':batch.input_ids.tolist()}
        dtype=next(e.parameters()).dtype;device=next(e.parameters()).device
        batch=batch.to(device,dtype if dtype!=t.float32 else None)
        assert e.architecture=='span'
        with t.inference_mode():
            h=e.encoder(input_ids=batch.input_ids,attention_mask=batch.attention_mask).last_hidden_state
            _,emb=e.processor.extract_embeddings_from_batch(h,batch.input_ids,batch)
            group=batch.task_types[0].index('classifications')
            logits=e.classifier(t.stack(emb[0][group])[1:]).squeeze(-1).float()
            p=t.softmax(logits,dim=-1).cpu().tolist()
        assert len(p)==len(opts)
        return dict(zip(labels(body),p)),{'candidate_logits':logits.cpu().tolist(),'input_tokens':batch.input_ids.shape[-1],
            'adapter':'full_state_questions_json_classification_not_native_noul'}

    def predict_openjev(self,body):
        t=self.torch;e=self.engine
        schema=e.build_schema(body['questions']);qs,fmt=schema['questions'],schema['format']
        assert len(qs)==1 and not schema['forced']
        # Matches OpenJev API text_of state serialization, not a pre-stringified benchmark state.
        prompt=e.chat_prompt_ids(e.system_text(qs,fmt),text(body['state']),thinking=False)
        template,slots=e.resolve_template(qs,fmt);canvas=e.build_canvas(template,slots,1234)
        n=len(prompt)+len(canvas);guard_count(n)
        native_cap=int(self.model.config.text_config.max_position_embeddings)
        if n>native_cap:raise InputTooLong(f'{n}>{native_cap}; no truncation')
        slot=slots[0]
        self.last_render={'prompt_token_ids':prompt,'canvas_token_ids':canvas,'schema':schema}
        with t.inference_mode():
            result=self.model(input_ids=t.tensor([prompt],device='cuda'),decoder_input_ids=t.tensor([canvas],device='cuda'))
            logits=result.logits[0,slot['pos'],slot['label_ids']].float()
            assert t.isfinite(logits).all().item()
            p=t.softmax(logits,dim=-1).cpu().tolist()
        return dict(zip(labels(body),p)),{'candidate_logits':logits.cpu().tolist(),'input_tokens':n,
            'adapter':'native_choice_noul_HF_eager','official_backend_parity_verified':False,'seed':1234}

    def close(self):
        if self.name=='openjev':asyncio.run(self.engine.close())
