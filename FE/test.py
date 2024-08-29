      
import src.inner_product.single_input_fe.elgamal_ip.nddf
import numpy as np
import unittest
import time

class TestElGamalInnerProduct(unittest.TestCase):

    def test_fin_result(self):

        fe =src.inner_product.single_input_fe.elgamal_ip.nddf
        pk, sk = fe.set_up(1024, 5)
        #TP
        pk1,sk1 = pk[0],sk[0]
        #print(sk1)
        #Encryptor
        pk2,sk2 = [pk[1],pk[2],pk[3]], [sk[1],sk[2],sk[3]]
        #print(pk2)
        #print(sk2)
        #decryptor
        pk3,sk3 = pk[4],sk[4]
        #mpk_2_i,msk_2_i=fe.set_up(1024,5)
        #mpk3,msk3=fe.set_up(1024,4)
        #print(mpk1,msk1)
        #pp=fe.output_p()
       

        y = [1, 1, 1]#y的个数要跟encrtyptor的个数一致，encryptor个数为3，y的长度也为3
        x = [1, 2, 3]#
        ctr=0
        aux='nddfe'
        skf = fe.KeyDerive(sk1,pk2,sk2,ctr,y,aux)
        #print(skf)
        t1=time.time()
        C_x = fe.Encrypt(pk1,sk2,pk3,ctr,x,aux)
        t2=time.time()
        t3=t2-t1
        print("加密的时间为",t3)
        #print(C_x)
        t4=time.time()
        result = fe.Decrypt(pk1,skf,sk3,C_x,y )
        t5=time.time()
        t6=t5-t4
        print("解密的时间为",t6)
        print("FE result is:", result)
       # obtained_inner_prod = fe.decrypt(pk, c_x, key_y, y, 2000)
        expected_inner_prod = np.inner(x, y)
        print("The correct result is:", expected_inner_prod)

        #try:
         #   assert obtained_inner_prod == expected_inner_prod
        #except AssertionError:
         #   print(
          #      f'The calculated inner product different than expected: {obtained_inner_prod} != {expected_inner_prod}')
       # print(f'The calculated inner product same as expected!: {obtained_inner_prod} == {expected_inner_prod}')

if __name__ == "__main__":
    unittest.main()

    

    

    