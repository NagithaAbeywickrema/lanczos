import matplotlib.pyplot as plt
import numpy as np

def read(fname: str) -> np.array:
    A = []
    with open(fname) as f:
        for line in f.readlines():
            A.append(line.split(","))
    return np.array(A)

def plot_bnd(A: np.array, title: str) -> None:
    kernels = np.unique(A[:, 0])
    fig = plt.figure(figsize=(9, 9))
    for kernel in kernels:     
        B = A[A[:, 0] == kernel]

        backends = np.unique(B[:, 1])
        kernels = np.unique(B[:, 0])
        
        for backend in backends:
            C = B[B[:, 1] == backend]
            words = C[:, 3].astype(int)
            time = C[:, 4].astype(float)
            if(title=="lanczos_lanczos_spmv_data"):
                valcount = C[:, 5].astype(float)
                words= valcount+valcount/2+ words+words/2+1/2
            
            bandwidthInGBs = ( (words)*8) / (10**9)
            plt.plot(words,bandwidthInGBs/time, label=f"{kernel}-backend-{backend}")

    plt.title(f"{title} Knl bnd", fontsize=23)
    plt.xlabel("Size", fontsize=18)
    plt.ylabel("Bandwidth (GB/sec)", fontsize=18)
    plt.xscale("log", base=10)
    plt.yscale("log", base=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig(f"{title}.pdf")
    plt.close()

def plot_all(A: np.array, title: str ,fun) -> None:
    kernels = np.unique(A[:, 0])
    
    for kernel in kernels:     
        B = A[A[:, 0] == kernel]

        backends = np.unique(B[:, 1])
        kernels = np.unique(B[:, 0])
        
        for backend in backends:
            C = B[B[:, 1] == backend]
            words = C[:, 3].astype(int)
            time = C[:, 4].astype(float)
            if(title=="lanczos_lanczos_spmv_data"):
                valcount = C[:, 5].astype(float)
                words= valcount+valcount/2+ words+words/2+1/2
            words=fun(words)
            bandwidthInGBs = ( (words)*8) / (10**9)
            if(title.__contains__("roofline")):
                plt.plot(words,bandwidthInGBs/time,linestyle='dashed', label=f"{kernel}-backend-{backend}")
            else:
                plt.plot(words,bandwidthInGBs/time, label=f"{kernel}-backend-{backend}")

def x4(x: int) -> int:
    return x*4

def x3(x: int) -> int:
    return x*3
def x2(x: int) -> int:
    return x*2
def mxm_vec(x: int) -> int:
    return x*x+x

def x1(x: int) -> int:
    return x


def plot(plot_graph,fun) -> None:
    fig = plt.figure(figsize=(9, 9))
    names={
           f"{plot_graph}":fun,
           "lanczos_roofline_data":x1,
           }
    for key in names.keys():
        name=key
        J = read(f"./build/{name}.txt")
        plot_all(J, name,names[key])

    plt.title(f"{plot_graph} Knl with roofline", fontsize=23)
    plt.xlabel("Words", fontsize=18)
    plt.ylabel("Bandwidth (GB/sec)", fontsize=18)
    plt.xscale("log", base=10)
    plt.yscale("log", base=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.savefig(f"with roofline {plot_graph}.pdf")
    plt.close()


if __name__ == "__main__":
    
    file_name="lanczos_lanczos_calc_w_data"
    # J = read(f"/home/pubudu/nomp/lanczos/build/{name}.txt")
    # plot_bnd(J, name)
    plot(file_name,x3)
