def add(a, b):
    return a + b


def minus(a, b):
    return a - b


def multi(a, b):
    return a * b


def divide(a, b):
    return a / b


import unittest


class TestFunc(unittest.TestCase):
    def setUp(self) -> None:
        print("do something before test stating")

    def tearDown(self) -> None:
        print("do something  after test ending")

    def test_add(self):
        print("add")
        self.assertEqual(3, add(1, 2))

    @unittest.skip("some reason")
    def test_minus(self):
        print("minus")
        self.assertEqual(1, minus(2, 1))

    @classmethod
    def setUpClass(cls):
        print("only called once")

    @classmethod
    def tearDownClass(cls) -> None:
        print("only called once when all test ending")


if __name__ == '__main__':
    # unittest.main(verbosity=2)

    # TestSuite
    # suite = unittest.TestSuite()
    # tests = [TestFunc("test_minus"), TestFunc("test_add")]
    # suite.addTest(tests)
    # runner = unittest.TextTestRunner(verbosity=2)
    # runner.run(suite)

    # TestLoader
    # unittest.TestLoader().loadTestsFromModule()
    # unittest.TestLoader().loadTestsFromName()
    # unittest.TestLoader().loadTestsFromNames()
    # unittest.TestLoader().loadTestsFromTestCase()
    # unittest.TestLoader().getTestCaseNames()
    # unittest.TestLoader().discover()

    # genrate test report
    # suite = unittest.TestSuite()
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunc))
    # with open("TextReport.txt", "w+") as f:
    #     runner = unittest.TextTestRunner(stream=f, verbosity=2)
    #     runner.run(suite)

    from HtmlTestRunner import HTMLTestRunner

    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunc))
    with open("HtmlReport.html", "w+") as f:
        runner = HTMLTestRunner(stream=f, report_title="test",
                                verbosity=2)
        runner.run(suite)
