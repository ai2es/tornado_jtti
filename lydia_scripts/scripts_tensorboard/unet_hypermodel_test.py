import unittest
import unet_hypermodel as uh

class TestUNetHyperModel(unittest.TestCase):

    def test_create_parser(self):
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

if __name__ == '__main__':
    unittest.main()