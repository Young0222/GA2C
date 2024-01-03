from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_smiles, from_smiles

# loaded_data = torch.load('data.pt')
# print("load_data: ", loaded_data)

# data_list = loaded_data.to_data_list()
# print("len(data_list): ", len(data_list))

# # 选择第一个图的数据
# single_data = data_list[1]
# print("single_data: ", single_data)
# single_data.x = single_data.x.int()

# smiles = to_smiles(single_data)
# print("smiles: ", smiles)

# # # 使用RDKit从SMILES创建分子对象
# molecule = Chem.MolFromSmiles(smiles)

# # # 将分子图保存为图片
# file_path = "mol.png"
# Draw.MolToFile(molecule, file_path)



##### from smiles #######

# CCOC(=O)c1cccc(Nc2c(-c3ccc(N(C)C)cc3)nc3cnccn32)c1
# O=C(O)CCN1C(=O)C(O)=C(C(=O)c2ccc(Cl)cc2)[C@@H]1c1ccc(Cl)cc1
# O=S(=O)(N[C@H]1C[C@@H]2CC[C@H]1C2)c1ccc(Br)s1
# Cc1ccc(S(=O)(=O)N2C[C@H](C(=O)N3CCOCC3)C3(C2)CCOCC3)cc1
# Cc1ccc(S(=O)(=O)CCC(=O)OCC(=O)Nc2cc(C)c(C)cc2[N+](=O)[O-])cc1

# CCOC(=O)c1cccc(Nc2c(-c3ccc(N(C)C)cc3)nc3cnccn32)c1
# CC(=O)NCCC1=CNc2c1cc(OC)cc2CC(=O)NCCc1c[nH]c2ccc(OC)cc12  褪黑激素
# CN1CCC[C@H]1c2cccnc2 尼古丁

# O=C1CC[C@H]2C(=O)NCCN12
# CCn1ccc2cc[nH]c(=O)c21

# smiles,zinc_id,inchikey,mwt,logp,reactive,purchasable,tranche_name,features
# Cc1ccc2c(c1)CNC2,ZINC000018531830,TZMYQGXSTSSHQV-UHFFFAOYSA-N,133.194,1.598,0,50,ADAA,
# C1=C(N2CCOCC2)CCC1,ZINC000000391801,VAPOFMGACKUWCI-UHFFFAOYSA-N,153.225,1.386,0,50,ADAA
# CC(C)C(Cl)=NOC(=O)Nc1ccccn1,ZINC000252621453,MYTSSGRGOOFIJD-UHFFFAOYSA-N,241.678,2.838,30,50,BFEA,
# CCn1ccc2cc[nH]c(=O)c21,ZINC000071373451,SETRUJSJLVLALL-UHFFFAOYSA-N,162.192,1.35,0,50,ADAA,
# COC(=O)c1cc(OC)cc(-c2cccc(F)c2)c1,ZINC000097757341,NPOHMNDKXVZAHP-UHFFFAOYSA-N,260.264,3.2880000000000003,0,50,CGAA,
# CCn1cc(NC(=O)Nc2ccc(Br)c(C)n2)cn1,ZINC000128684281,BEWHKEZDKWSIJE-UHFFFAOYSA-N,324.182,3.013,0,50,DGAA,


smiles = 'CCn1cc(NC(=O)Nc2ccc(Br)c(C)n2)cn1'
file_path = "6.png"
molecule = Chem.MolFromSmiles(smiles)
Draw.MolToFile(molecule, file_path, size=(900, 900))

data = from_smiles(smiles)
print("Data: ", data)


