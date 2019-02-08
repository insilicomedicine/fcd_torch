from fcd_torch import FCD
import unittest


class test_same_output(unittest.TestCase):
    def test_output(self):
        fcd = FCD()
        set1 = ['Oc1ccccc1-c1cccc2cnccc12',
                'COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1']
        set2 = ['CNC', 'CCCP',
                'Oc1ccccc1-c1cccc2cnccc12',
                'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1',
                'Cc1nc(NCc2ccccc2)no1-c1ccccc1']
        output_keras = 52.83132961802335
        output_pytorch = fcd(set1, set2)
        diff = abs(output_keras-output_pytorch)
        self.assertAlmostEqual(
            output_pytorch, output_keras, places=4,
            msg=("Outputs differ. keras={},".format(output_keras) +
                 "torch={}. diff is {}".format(output_pytorch, diff))
        )


if __name__ == '__main__':
    unittest.main()
