"""Release images pass automatically without bypassing secret checks."""
from pathlib import Path
import sys
import tempfile
import unittest
import zlib
import struct

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scan_release import scan


def png(metadata=b''):
    def chunk(kind,data):
        return (struct.pack('>I',len(data))+kind+data
                +struct.pack('>I',zlib.crc32(kind+data)))
    return (b'\x89PNG\r\n\x1a\n'
            +chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))
            +chunk(b'tEXt',b'Comment\x00'+metadata)
            +chunk(b'IDAT',zlib.compress(b'\x00\x00\x00\x00'))
            +chunk(b'IEND',b''))


class ReleaseScanTest(unittest.TestCase):
    def scan_file(self,name,data):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
            return scan(root)

    def test_presentation_png_passes(self):
        self.assertTrue(self.scan_file('assets/banner.png',png())['clean'])

    def test_png_metadata_secrets_are_detected(self):
        result=self.scan_file('assets/banner.png',png(b'sk-'+b'A'*30))
        self.assertIn('api_key',{issue['kind'] for issue in result['issues']})

    def test_other_binaries_still_require_review(self):
        for name,data in [('assets/fake.png',b'\xffnot a PNG'),
                          ('assets/truncated.png',png()[:-12]),
                          ('payload.png',png()),('assets/payload.bin',png())]:
            with self.subTest(name=name):
                result=self.scan_file(name,data)
                self.assertIn('binary_requires_review',
                              {issue['kind'] for issue in result['issues']})

    def test_text_secrets_are_detected(self):
        self.assertFalse(self.scan_file('config.txt',b'hf_'+b'A'*30)['clean'])

    def test_forbidden_names_are_detected(self):
        self.assertFalse(self.scan_file('.env',b'EXAMPLE=value')['clean'])


if __name__=='__main__':unittest.main()
