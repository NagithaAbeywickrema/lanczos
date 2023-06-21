import loopy as lp
from loopy.symbolic import Reduction
import pymbolic.primitives as prim

LOOPY_LANG_VERSION = (2018, 2)

def stream_data_flow_loop(knl, context):
    (iname,) = knl.default_entrypoint.all_inames()
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    knl = lp.split_iname(
        knl, iname, 32, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
    return knl

def mat_vec_mul(knl, context):
    knl = lp.split_iname(
        knl, "i", 32, inner_iname="i_inner", outer_iname="i_outer"
    )
    knl = lp.tag_inames(knl, {"i_outer": "g.0", "i_inner": "l.0", "k": "for"})
    return knl

def spmv(knl, context):
    knl = lp.split_iname(
        knl, "row", 32, inner_iname="row_inner", outer_iname="row_outer"
    )

    knl = lp.tag_inames(knl, {"row_outer": "g.0", "row_inner": "l.0", "jj": "for"})
    return knl

def vector(knl, context):
    (i,) = knl.default_entrypoint.all_inames()
    knl = lp.split_iname(knl, i, 32, outer_tag="g.0", inner_tag="l.0")
    return knl
