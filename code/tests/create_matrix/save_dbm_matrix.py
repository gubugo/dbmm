
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

from code.utils.metrics import metric_distance_to_nearest_neighbor
from code.utils.scatterplot import plot_decision_map_with_points, plot_decision_map_with_points_relative

viridis_cmap = plt.get_cmap("viridis")

def alpha_function(x):
    return (1/(1+np.exp(3*x-17)))**4

class dbm_saver():
    def __init__(self):
        self.confidence = []
        self.dntp_map = []
        self.dntp_matrix = []

    def flush_variables(self):
        self.confidence = []
        self.dntp_map = []
        self.dntp_matrix = []

    def show_single_dbm(self, dbm, ax):
        ax.imshow(
            dbm,
            origin="lower",
            interpolation="none",
            resample=False,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off") 

    def show_dbm(self, dbm, ax):
        self.show_single_dbm(dbm, ax)

    def show_dbm_scatterplot(self, dbm, x, y, grid_res, ax):
        self.show_single_dbm(dbm, ax)
        plot_decision_map_with_points(x, y, grid_res, ax)

    def show_dbm_local_scatterplot(self, dbm, inverted_grid, nd_data, x, y, grid_res, ax):
        self.show_single_dbm(dbm, ax)

        invp_grid_neighbor_finder = NearestNeighbors(
            n_neighbors=5
        ) 
        invp_grid_neighbor_finder.fit(inverted_grid)
        alpha = np.zeros((np.shape(nd_data)[0],))

        values = metric_distance_to_nearest_neighbor(nd_data, invp_grid_neighbor_finder)
        for index, value in enumerate(values):
            alpha[index] = alpha_function(value)

        plot_decision_map_with_points_relative(x, y, alpha, grid_res, ax)

    def show_class_confidence_map(self, classifier, inverted_grid, grid_res, ax):
        if len(self.confidence) == 0:
            res = classifier.predict_proba(inverted_grid)
            
            confidence = np.zeros(np.shape(res)[0])

            for k,lis in enumerate(res):
                confidence[k] = np.max(lis)

            self.confidence = confidence

        self.show_single_dbm(viridis_cmap(self.confidence).reshape((grid_res, grid_res, 4)), ax)


    def show_class_confidence_dbm(self, dbm, classifier, inverted_grid, grid_res, ax):
        conf_dbm = np.zeros((grid_res,grid_res,4))
        
        if len(self.confidence) == 0:
            res = classifier.predict_proba(inverted_grid)
            
            confidence = np.zeros(np.shape(res)[0])

            for k,lis in enumerate(res):
                confidence[k] = np.max(lis)

            self.confidence = confidence

        confidence = self.confidence.reshape(grid_res,grid_res)
        conf_dbm[:,:,0] = dbm[:,:,0]*confidence
        conf_dbm[:,:,1] = dbm[:,:,1]*confidence
        conf_dbm[:,:,2] = dbm[:,:,2]*confidence
        conf_dbm[:,:,3] = dbm[:,:,3]

        self.show_single_dbm(conf_dbm, ax)

    def show_distance_to_nearest_training_point_map(self, neighbor_finder, inverted_grid, grid_res, ax):
        if len(self.dntp_map) == 0:
            self.dntp_map = metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)
        minmaxed_dntp = 1.0-minmax_scale(self.dntp_map)

        self.show_single_dbm(viridis_cmap(minmaxed_dntp).reshape((grid_res, grid_res, 4)), ax)

    def show_distance_to_nearest_training_point_dbm(self, dbm, neighbor_finder, inverted_grid, grid_res, ax):
        ntp_dbm = np.zeros((grid_res,grid_res,4))
        if len(self.dntp_map) == 0:
            self.dntp_map = metric_distance_to_nearest_neighbor(inverted_grid, neighbor_finder)
        minmaxed_dntp = 1.0-minmax_scale(self.dntp_map)

        minmaxed_dntp = minmaxed_dntp.reshape(grid_res,grid_res)
        ntp_dbm[:,:,0] = dbm[:,:,0]*(minmaxed_dntp)
        ntp_dbm[:,:,1] = dbm[:,:,1]*(minmaxed_dntp)
        ntp_dbm[:,:,2] = dbm[:,:,2]*(minmaxed_dntp)
        ntp_dbm[:,:,3] = dbm[:,:,3]

        self.show_single_dbm(ntp_dbm, ax)

    def show_global_distance_to_nearest_training_point_map(self, dntp_matrix_tobe_scaled, matrix_coord, grid_res, ax):
        if len(self.dntp_matrix) == 0:
            self.dntp_matrix = 1.0-minmax_scale(dntp_matrix_tobe_scaled)
        viridis_cmapped = viridis_cmap(self.dntp_matrix)

        self.show_single_dbm(viridis_cmapped[matrix_coord].reshape((grid_res, grid_res, 4)), ax)

    def show_global_distance_to_nearest_training_point_dbm(self,dbm_matrix, dntp_matrix_tobe_scaled, matrix_coord, grid_res, ax):
        if len(self.dntp_matrix) == 0:
            self.dntp_matrix = 1.0-minmax_scale(dntp_matrix_tobe_scaled)

        dbm_matrix[matrix_coord,:,0] = dbm_matrix[matrix_coord,:,0]*self.dntp_matrix[matrix_coord]
        dbm_matrix[matrix_coord,:,1] = dbm_matrix[matrix_coord,:,1]*self.dntp_matrix[matrix_coord]
        dbm_matrix[matrix_coord,:,2] = dbm_matrix[matrix_coord,:,2]*self.dntp_matrix[matrix_coord]
        dbm_matrix[matrix_coord,:,3] = dbm_matrix[matrix_coord,:,3]

        self.show_single_dbm(dbm_matrix[matrix_coord].reshape((grid_res, grid_res, 4)), ax)

