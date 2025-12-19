import numpy as np
from grakel.kernels import WeisfeilerLehman
from grakel.kernels import VertexHistogram
from grakel.kernels import EdgeHistogram
from grakel.kernels import ShortestPath
from grakel.kernels import GraphletSampling

class GraphkernelFunc():
    def __init__(self, kernelType, kernelParam=1):
        self.kernelType = kernelType
        self.kernelParam = kernelParam

        # カーネル関数の設定
        kernelFuncs = [self.k_sp, self.k_gs, self.k_WLsubtree,self.k_vh,self.k_eh]
        self.createMatrix = kernelFuncs[kernelType]
    
    #-----------------------------------------------------------------------------------------#
    # 最短経路カーネル
    def k_sp(self,graphs):
        # ShortestPath カーネルの初期化
        sp_kernel = ShortestPath(with_labels=False)
        # カーネル計算
        kernel_matrix = sp_kernel.fit_transform(graphs)
        return kernel_matrix
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # グラフレットカーネル
    def k_gs(self,graphs):
        # GraphletSamplingカーネルの初期化
        gs_kernel = GraphletSampling(n_jobs=6,
                                     normalize=False, 
                                     verbose=False,
                                     random_state=None,
                                     k=5,
                                     sampling={"n_samples" : 50})  
        #gs_kernel = GraphletSampling({"sampling":10})  # サンプリング回数を指定
        # カーネル計算
        kernel_matrix = gs_kernel.fit_transform(graphs)
        return kernel_matrix
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # Weisfeiler-Lehman部分木カーネル
    def k_WLsubtree(self,graphs,i):
        # Weisfeiler-Lehmanカーネルのインスタンス化
        wl_kernel = WeisfeilerLehman(n_iter=i, normalize=True)
        # カーネルの計算
        kernel_matrix = wl_kernel.fit_transform(graphs)
        return kernel_matrix
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # 頂点ヒストグラムカーネル
    def k_vh(self,graphs):
        # Vertex Histogram カーネルの初期化
        vh_kernel = VertexHistogram()
        # カーネル計算
        kernel_matrix = vh_kernel.fit_transform(graphs)
        return kernel_matrix
    #-----------------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------------#
    # 辺ヒストグラムカーネル
    def k_eh(self,graphs):
        # Edge Histogram カーネルの初期化
        eh_kernel = EdgeHistogram()
        # カーネル計算
        kernel_matrix = eh_kernel.fit_transform(graphs)
        return kernel_matrix
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # グラム行列の正規化
    @staticmethod
    def normalize_gram_matrix(gram_matrix):
        n = gram_matrix.shape[0]
        gram_matrix_norm = np.zeros([n, n], dtype=np.float32)

        for i in range(0, n):
            for j in range(i, n):
                if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                    g = gram_matrix[i][j] / np.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                    gram_matrix_norm[i][j] = g
                    gram_matrix_norm[j][i] = g

        return gram_matrix_norm
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # 頂点カーネル graph -> scalar
    @staticmethod
    def k_func_vh(g1, g2):
        gk = VertexHistogram()
        gk.fit([g1])
        k_val = gk.transform([g2])[0]
        return k_val
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # Weisfeiler-Lehman部分木カーネル graph -> scalar
    @staticmethod
    def k_func_wl(g1, g2, i):
        gk = WeisfeilerLehman(n_iter=i, normalize=True)
        gk.fit([g1])
        k_val = gk.transform([g2])[0]
        return k_val
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    # Weisfeiler-Lehman部分木カーネル graph -> vector
    @staticmethod
    def k_vec_wl(g1, g2, i):
        gk = WeisfeilerLehman(n_iter=i, normalize=True)
        gk.fit([g1])
        k_val = (gk.transform(g2).T)[0]
        return k_val
    #-----------------------------------------------------------------------------------------#



    