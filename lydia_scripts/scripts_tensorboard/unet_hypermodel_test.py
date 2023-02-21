import sys, unittest, argparse
import unet_hypermodel as uh

class TestUNetHyperModel(unittest.TestCase):

    def test_create_parser(self):
        argv = sys.argv[1:]
        args = ['--in_dir', '../../../test_data',
                '--out_dir', '../../../test_data']
        parser = uh.create_parser()
        self.assertEqual('foo'.upper(), 'FOO')

    def test_parse_args(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_args2string(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_init(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_build(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_create_tuner(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

def parse_args():
    parser = uh.create_parser()
    ns, args = parser.parse_known_args(namespace=unittest)
    print(ns, args)
    return ns, sys.argv[:1] + args

if __name__ == '__main__':
    args, unittest_args = parse_args()   # run this first
    print(args, unittest_args)
    sys.argv[:] = unittest_args       # create cleans argv for main()
    unittest.main()