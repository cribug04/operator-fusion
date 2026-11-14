"""模拟 dort 的进程与输出文件，只测试脚本控制流，不模拟 NPU 计算。"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAKE = '''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
mode = os.environ['FAKE_MODE']
if args[0] == 'device':
    print('OS DOWN RT up' if mode == 'unhealthy' else 'OS up RT up')
    sys.exit(0)
if mode == 'run-error':
    sys.exit(3)
output = Path(args[args.index('--raw-out')+1])
model = args[args.index('-m')+1]
size = {'resize_pkg':131072,'resize_big_pkg':2097152,'conv_s2_pkg':903168,'conv_s2_ref_pkg':903168}[model]
if mode == 'missing':
    sys.exit(0)
if mode == 'truncated':
    size -= 2
bits = b'\\x80\\x3f'
if mode == 'nondeterministic' and 'run2' in output.name:
    bits = b'\\x00\\x40'
if mode == 'wrong-values':
    bits = b'\\x00\\x40'
output.write_bytes(bits*(size//2))
'''
HASH = '''#!/usr/bin/env python3
import hashlib, sys
from pathlib import Path
for name in sys.argv[1:]:
    print(hashlib.sha256(Path(name).read_bytes()).hexdigest(), name)
'''

class BoardRunnerTests(unittest.TestCase):
    def run_case(self, mode, expected_success, expected_message):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bins = root/'bin'
            bins.mkdir()
            for name, content in [('dort', FAKE), ('sha256sum', HASH)]:
                p = bins/name
                p.write_text(content)
                p.chmod(0o755)
            for case, size in [('resize_small',131072),('resize_big',2097152),('conv_s2',819200)]:
                folder = root/'cases'/case
                folder.mkdir(parents=True)
                (folder/'expected.bin').write_bytes(b'\x80\x3f'*(size//2))
            env = dict(os.environ, FAKE_MODE=mode, DORT=str(bins/'dort'), PY=sys.executable,
                       CASE_DIR=str(root/'cases'), OUT=str(root/'out'), CAL_ROOT=str(root/'cal'),
                       PATH=str(bins)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
            result = subprocess.run(['bash',str(ROOT/'onboard/run_on_board.sh')],env=env,capture_output=True,text=True,timeout=30)
            text = result.stdout + result.stderr
            self.assertEqual(result.returncode == 0, expected_success, text)
            self.assertIn(expected_message, text)

    def test_successful_control_flow(self):
        self.run_case('ok',True,'PASS: 四个用例')

    def test_health_failure(self):
        self.run_case('unhealthy',False,'board not ready')

    def test_process_failure(self):
        self.run_case('run-error',False,'运行失败')

    def test_missing_output(self):
        self.run_case('missing',False,'没有生成输出')

    def test_truncated_output(self):
        self.run_case('truncated',False,'输出字节数不正确')

    def test_nondeterministic_output(self):
        self.run_case('nondeterministic',False,'三次输出不一致')

    def test_numerically_wrong_output(self):
        self.run_case('wrong-values',False,'VERDICT: DIFF')

if __name__ == '__main__':
    unittest.main()
