import loopy as lp
from loopy.symbolic import Reduction
import pymbolic.primitives as prim

LOOPY_LANG_VERSION = (2018, 2)

def custom_optimize_reduction(
    tunit: lp.translation_unit.TranslationUnit
) -> lp.translation_unit.TranslationUnit:
    """Perform transformations to realize a reduction."""

    if len(tunit.callables_table.keys()) > 1:
        raise NotImplementedError(
            "Don't know how to handle more than 1 callable in translation unit!"
        )

    knl = tunit.default_entrypoint

    (knl_name,) = tunit.callables_table.keys()

    knl, insns = tunit[knl_name], []

    for i in range(1,4):
      ins1 = knl.instructions[i]
      rhs1 = ins1.expression
      lhs1 = ins1.assignee 
      x=lp.Assignment(lhs1, rhs1)
      insns.append(x)
    
    ins = knl.instructions[4]
    (lhs, *rhs) = ins.expression.children
    rhs = Reduction(lp.library.reduction.SumReductionOperation(), "jj", prim.Sum(tuple(rhs)))
    ins5 = knl.instructions[5]
    (lhs5, *rhs5) = ins5.expression.children
    x4 = lp.Assignment(lhs5, rhs)
    insns.append(x4)

    tv = list(knl.temporary_variables.values())
    tunit = lp.make_kernel(
        knl.domains,
        insns,
        knl.args+tv[1:],
        name=knl.name,
        target=knl.target,
        lang_version=LOOPY_LANG_VERSION, 
    )
    tunit = lp.tag_inames(tunit, {"row_outer": "g.0"})
    tunit = lp.tag_inames(tunit, {"row_inner": "l.0"})
    
    return tunit

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

    knl = custom_optimize_reduction(knl)
    return knl
