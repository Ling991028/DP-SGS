'''
所有py文件统一的eit数据参数配置
'''

from pyeit.mesh import create, set_perm
import pyeit.eit.jac as jac
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import thorax, circle

n_el, h0 = 16, 0.03
background = 1.0
margin = 0.05
el_dist, step = 1, 1
seed = 2022
mesh_obj, el_pos = create(seed, n_el=n_el, fd=circle, h0=h0)
fwd = Forward(mesh_obj, el_pos)
ex_mat = eit_scan_lines(n_el, el_dist)

