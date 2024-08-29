      
from charm.toolbox.integergroup import IntegerGroupQ, integer
from typing import List, Dict, Tuple
from src.helpers.additive_elgamal import AdditiveElGamal, ElGamalCipher
from src.helpers.helpers import reduce_vector_mod, get_int
from src.errors.wrong_vector_for_provided_key import WrongVectorForProvidedKey
import charm
import numpy as np
import hashlib
#from src.helpers import dummy_discrete_log, get_int, get_modulus

IntegerGroupElement = charm.core.math.integer.integer
ElGamalKey = Dict[str, IntegerGroupElement]
#生成群
debug = True
p = integer(
    148829018183496626261556856344710600327516732500226144177322012998064772051982752493460332138204351040296264880017943408846937646702376203733370973197019636813306480144595809796154634625021213611577190781215296823124523899584781302512549499802030946698512327294159881907114777803654670044046376468983244647367)
q = integer(
    74414509091748313130778428172355300163758366250113072088661006499032386025991376246730166069102175520148132440008971704423468823351188101866685486598509818406653240072297904898077317312510606805788595390607648411562261949792390651256274749901015473349256163647079940953557388901827335022023188234491622323683)
elgamal_group = IntegerGroupQ()
elgamal = AdditiveElGamal(elgamal_group, p, q)
elgamal_params = {"group": elgamal_group, "p": int(p)}
#输出群
def output_p():
    return {"group": elgamal_group, "p": int(p)}  
#创建哈希函数
def H1(data: str, p: integer) -> integer:
    hash_object = hashlib.sha256(data.encode('utf-8'))
    return integer(int(hash_object.hexdigest(), 16) % p)


#生成公钥密钥
def set_up(security_parameter: int, vector_length: int) -> Tuple[List[ElGamalKey], List[ElGamalKey]]:
  
    master_public_key = [None] * vector_length
    master_secret_key = [None] * vector_length
    for i in range(vector_length):
        (master_public_key[i], master_secret_key[i]) = elgamal.keygen(secparam=security_parameter)
    return master_public_key, master_secret_key
#生成功能密钥
def KeyDerive(master_secret_key: List[ElGamalKey], master_public_key2:List[ElGamalKey], master_secret_key2: List[ElGamalKey], ctr: int, y: List[int], aux: str) -> integer:
    y = reduce_vector_mod(y, elgamal_params['p'])   # y的个数要跟encrtyptor的个数一致，encryptor个数为3，y的长度也为3
    ctr += 1      # 计数器增加1。
    skf = integer(0)# 初始化一个'skf'，初始值为整数0

    for i in range(len(y)):
        sk2i = master_secret_key2[i] # 获取次级公钥列表中的第i个公钥。
        pk2i = master_public_key2[i]

        
        #if 'x' not in pk2i: # 如果该次级公钥缺少'x'分量，则抛出错误。
            #raise ValueError(f"The public key at index {i} is missing the 'x' component.")
        
        #print(pk2i['h'])
        #print(master_secret_key[0]['x'] * sk2i['x'])
        #print("this the secret key of tp")
        #print(master_secret_key['x'])
        key = pk2i['h'] ** (master_secret_key['x'])
        #print(key)
        r = H1(f"{key}{ctr}{aux}", elgamal_params['p']) # pk2i的'g'分量的master_secret_key次方，pk2i的'x'分量，计数器ctr，辅助字符串aux，以及模数p。
        skf += r * y[i]
    print(type(skf))
    
    return reduce_vector_mod([skf], elgamal_params['p'])[0]# 返回通过'reduce_vector_mod'函数将密钥因子模p约减以确保其在正确的密钥空间范围内的结果。
    

def Encrypt(master_public_key: List[ElGamalKey], master_secret_key2: List[ElGamalKey],master_public_key3:List[ElGamalKey],ctr: int,  X : List[int], aux: str)-> integer:
    C = []
    ctr += 1 
    for i in range(len(X)):
        sk2i = master_secret_key2[i] # 获取次级公钥列表中的第i个公钥。
        key_2=(master_public_key['h'])**sk2i["x"]
        r= H1(f"{key_2}{ctr}{aux}", elgamal_params['p'])
        C_i=((master_public_key['h'])**r)*(master_public_key3['h']**X[i])
        C.append(C_i)
    print(type(C))
        
    return (C) 

def Decrypt(master_public_key: List[ElGamalKey],skf:List[ElGamalKey],master_secret_key3:List[ElGamalKey],C:List[ElGamalKey],y:List[int]) -> integer:
    #C= reduce_vector_mod(C, elgamal_params['p'])
    y = reduce_vector_mod(y, elgamal_params['p'])#将y,c的长度固定一样
    #print(C,y)
    sum1= integer(1)
    E = integer(0)
    #print(sum1)
    #for i  in range(len(y)):
     #   print(C[i])
      #  print(y[i])
       # sum1 =sum1* (C[i]**y[i])
    #sum2=1/(master_public_key['h']**skf)    
    #print(sum2)
    #print(sum1)
    #sum1= reduce_vector_mod([sum1], elgamal_params['p'])
    #sum2= reduce_vector_mod([sum2], elgamal_params['p'])
    #print(type(sum1),type(sum2))
    #for i in range(len(y)): 
    #    E = E+sum1[i]*sum2[i]
    #E = sum2[0] * sum1[0]

    #print(E)


    c2 = np.product([C[i] ** y[i] for i in range(len(C))])
    #print("c2:",c2)
    sum2=1/(master_public_key['h']**skf)
    E = c2 * sum2
    #print(1/master_secret_key3['x'])
    
    #get g^<x,y>
    E = pow(E,1/master_secret_key3['x'])#E**(1/master_secret_key3['x'])
    
    #get <x,y>
    result = dummy_discrete_log(master_public_key['g'], E, elgamal_params['p'], 200)
    #print(E)
    return result       



def dummy_discrete_log(a: int, b: int, mod: int, limit: int) -> int:
    """Calculates discrete log of b in the base of a modulo mod, provided the
    result is smaller than limit. Otherwise, returns None

    Args:
        a (int): base of logarithm
        b (int): number from which the logarithm is calculated
        mod (int): modulus of logarithm 
        limit (int): limit within which the result should lie

    Returns:
        int: result of logarithm or None if the result was not found withn the limit
    """
    for i in range(limit):
        if pow(a, i, mod) == b:
            return i
    return None