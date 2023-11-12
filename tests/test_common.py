from unittest import TestCase

from texmo.common import itoa3


class CommonTest(TestCase):

    def test_itoa3(self):
        self.assertEqual(itoa3(0), "0")
        self.assertEqual(itoa3(1), "1")
        self.assertEqual(itoa3(900), "900")
        self.assertEqual(itoa3(999), "999")
        self.assertEqual(itoa3(1000), "1000")
        self.assertEqual(itoa3(1800), "1800")
        self.assertEqual(itoa3(1999), "1999")
        self.assertEqual(itoa3(2000), "2.00k")
        self.assertEqual(itoa3(2001), "2.00k")
        self.assertEqual(itoa3(2010), "2.01k")
        self.assertEqual(itoa3(12345), "12.35k")
        self.assertEqual(itoa3(65536), "65.5k")
        self.assertEqual(itoa3(123456), "123.5k")
        self.assertEqual(itoa3(500000), "500k")
        self.assertEqual(itoa3(1234567), "1235k")
        self.assertEqual(itoa3(1999999), "2.00M")
        self.assertEqual(itoa3(2345678), "2.35M")
        

