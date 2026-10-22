"""Thin adapters for checkpoint-provided PPLX code and official OneJev/qev."""
import hashlib
import json
import os
import sys
from pathlib import Path
from common import digest

class Adapter:
    def __init__(self,name,path):
        import torch
        self.torch=torch; self.name=name; self.last_render={}; path=Path(path)
        sys.path.insert(0,str(path/'source/src'))
        torch.set_num_threads(4)
        if name=='pplx-v11':
            from autojev.model import DecisionModel,answer,decision_messages,options
            self.engine=DecisionModel(path,device='cuda:0',cpu_threads=4)
            self.answer=answer; self.messages=decision_messages; self.options=options
            assert self.engine.attention_mode=='noncausal_full_attention' and self.engine.pooling=='last'
        elif name=='pplx-v12':
            from pplx_decider import Decider,options
            self.engine=Decider.from_pretrained(path,device='cuda:0',fast=False,cpu_threads=4)
            self.options=options
            assert self.engine.fast is None and self.engine.max_length==262144
        elif name=='qev':
            source=Path(os.environ['MEDBENCH_UPSTREAM_SOURCE'])
            sys.path.insert(0,str(source))
            from qev.engine import DecisionEngine
            self.engine=DecisionEngine(str(path),device='cuda:0',max_branch_tokens=16384)
            assert self.engine.readout.config.attention=='noncausal'
            assert self.engine.readout.config.readout=='slot_lm'
        else: raise ValueError(name)
        if name.startswith('pplx'):
            assert self.engine.temperature==json.loads((path/'decision_config.json').read_text())['temperature']

    def predict(self,body):
        q=body['questions']['decision']; e=self.engine
        if self.name.startswith('pplx'):
            labels,descriptions=self.options(q)
            assert labels==(list(q['criteria']) if q['type']=='choice' else ['false','true'])
            if q['type']=='noul': assert descriptions==[q['criteria']['false'],q['criteria']['true']]
        if self.name=='pplx-v11':
            row={'state':body['state'],'question':q}
            prompt=e.processor.apply_chat_template(self.messages(row,e.codes),tokenize=False,
                          add_generation_prompt=True,enable_thinking=False)
            ids=e.processor.tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)>8192: raise ValueError('context_length: exceeds 8192; no truncation')
            ps=e.predict([row],batch_size=1)[0]
            response={'answers':{'decision':self.answer(q,ps)}}
            self.last_render={'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                              'input_token_ids_sha256':digest(ids),'input_tokens':len(ids),
                              'native_probabilities':ps,'temperature':e.temperature}
        elif self.name=='pplx-v12':
            # Observe the exact native encoding, without an additional model forward.
            original=e.encoder.encode; trace={}
            def capture(request):
                encoded=original(request); prompt=e.encoder.text(request)
                ids=encoded.input_ids.tolist()
                assert e.processor.tokenizer.encode(prompt,add_special_tokens=False)==ids
                trace.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                             input_token_ids_sha256=digest(ids),input_tokens=encoded.length,
                             markers=encoded.markers.tolist(),temperature=e.temperature)
                return encoded
            e.encoder.encode=capture
            try: response=e.decide(body)
            finally: e.encoder.encode=original
            self.last_render=trace
        else:
            from qev.schema import SystemOneRequest
            from qev.prompt import render_question
            request=SystemOneRequest(state=body['state'],questions=body['questions'])
            assert request.questions['decision'].model_dump(exclude_none=True)==q
            rendered=render_question('decision',request.questions['decision'],style=e.style)
            assert rendered.labels==(list(q['criteria']) if q['type']=='choice' else ['yes','no'])
            prompt=e._render_prompt(request.state,rendered.suffix); ids=e._encode(prompt)
            response,meta=e.decide(request,debias=1,debug=True)
            response=response.model_dump()
            self.last_render={'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                              'input_token_ids_sha256':digest(ids),'input_tokens':len(ids),'native_metadata':meta}
            assert meta['branches']==1 and meta['prompt_sha256']==self.last_render['prompt_sha256']
        return response

    def close(self): pass
