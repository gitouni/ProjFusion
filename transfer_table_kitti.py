from collections import OrderedDict
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir",type=str,default="log/kitti")
parser.add_argument("--method_list",type=str,nargs="+",default=['calibnet','rggnet','lccnet','lccnet_mr5','lccraft','calibdepth','pool_3c_t_bc','pool_3c_t','pool_3c_t_res_bc','pool_3c_t_res','pool_3c_t_s15'])
parser.add_argument("--name_method_list",type=str,nargs="+",default=['CalibNet','RGGNet','LCCNet','LCCNet (mr5)','LCCRAFT','CalibDepth',
                         'RLCalib (IL)','RLCalib (IL+RL)','RLCalib (IL) (res)' ,'RLCalib (IL+RL) (res)', 'RLCalib (IL+RL) (s15)'])
parser.add_argument("--first_suffix",type=str,nargs="+",default=[r'~\cite{CalibNet}',r'~\cite{RGGNet}',r'~\cite{LCCNet}','',r'~\cite{LCCRAFT}',r'~\cite{CalibDepth}','','','','',''])
parser.add_argument("--key_order",type=str,nargs="+",default=['Rx','Ry','Rz','RRMSE','RMAE',
                                                              'tx','ty','tz','tRMSE','tMAE','3d3c','5d5c'])
parser.add_argument("--save_table",type=str,default="tmp_table_kitti.txt")
parser.add_argument("--row_head",type=str,default="~ &")
args = parser.parse_args()
with open(args.save_table,'w') as f:
    for i, (method, method_name) in enumerate(zip(args.method_list, args.name_method_list)):
        filename = os.path.join(args.src_dir, method+'.json')
        assert os.path.isfile(filename), "{} does not exist".format(filename)
        summary = json.load(open(filename,'r'))[-1]  # summary in the last dict
        values = []
        f.write(args.row_head)
        f.write(method_name)
        f.write(args.first_suffix[i])
        for key in args.key_order:
            value = summary[key]
            if 't' in key:
                value *= 100
                values.append("{:.3f}".format(value))
            elif key == '3d3c' or key == '5d5c':
                value *= 100
                values.append("{:.2f}".format(value)+r'\%')
            else:
                values.append("{:.3f}".format(value))
        f.write(r" &"+r" &".join(values))
        f.write(r" \\")
        f.write('\n')
    f.write(r'\bottomrule')
print('Table saved to {}'.format(args.save_table))