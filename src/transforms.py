import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def matrix_norm(knl):
    (i,) = knl.default_entrypoint.all_inames()
    knl = lp.split_iname(knl, i, 32)
    knl = lp.tag_inames(
        knl, [(f"{i}_outer", "g.0"), (f"{i}_inner", "l.0")]
    )
    return knl

def matrix_mul(knl):
    knl = lp.tag_inames(knl, [("i", "g.0"), ("j", "g.1")])
    return knl

def identity_mtx(knl):
    (g0, g1) = knl.default_entrypoint.all_inames()
    knl = lp.tag_inames(knl, [(g0, "g.0"), (g1, "g.1")])
    return knl

def qr_algo(knl):
    knl = lp.prioritize_loops(knl, "i,j")
    knl = lp.tag_inames(knl, [("i", "g.0"), ("j", "g.1")])
    return knl
