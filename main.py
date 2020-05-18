#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wuyin


import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from PIL import Image


class Settings:
    def __init__(self):
        # image load
        self.image = Image.open('data.png')
        self.data = np.array(self.image)[:, :, :3].astype(np.float) / 255

        # Parameters initialize
        self.k_init = 10  # the initial number of cluster
        self.n_min = 10000  # the minimum number of points per cluster
        self.i_max = 10  # the number of iterations
        self.sigma_max = 0.1  # the maximum standard deviation of points within a cluster
        self.L_min = 0.2  # the maximum distance between c
        self.P_max = 1  # the maximum number of clusters that can be merged per iteration

        # parameters of the image
        self.width = self.data.shape[0]
        self.height = self.data.shape[1]
        self.pixels = self.width * self.height


def clu_center_select(settings, cluster):
    # 1.
    # Set k_init and randomly select k clusters centers Z from S
    print('---Step 1: cluster center initialize')

    z = np.empty([cluster, 3])
    cluster_select = 0
    while cluster_select < cluster:
        imx = random.randint(0, settings.width - 1)
        imy = random.randint(0, settings.height - 1)
        imcolor = settings.data[imx, imy, :]
        repeat = False

        # if the selected color is repeat, then give it up
        for k in range(cluster):
            if (z[k] == imcolor).all():
                print("repeat")
                repeat = True
                break

        if repeat is False:
            z[cluster_select] = np.copy(imcolor)
            cluster_select += 1

    print('finish')

    return z


def assign_points(settings, cluster, z):
    # 2.
    # assign each point to its closest cluster center
    print('---Step 2: assigning the Set')

    # distances
    dists = np.empty([cluster, settings.width, settings.height])
    s = [[] for i in range(cluster)]

    # calculate the distance of each center between z[k] and each point x[i, j]
    for k in range(cluster):
        for i in range(settings.width):
            for j in range(settings.height):
                dists[k, i, j] = np.linalg.norm(settings.data[i, j, :] - z[k, :])

    len_sum = 0
    # find every minimum distance center, and collect them into corresponding set S[k], it comprise the position.
    for i in range(settings.width):
        for j in range(settings.height):
            for k in range(cluster):
                if dists[k, i, j] == np.amin(dists[:, i, j]):
                    s[k].append([i, j])
                    len_sum += 1
                    break

    print('finish')

    return s


def remove_cluster(settings, cluster, z, s):
    # 3.
    # remove the cluster with too little points
    print('---Step 3: Removing......')

    remove_mark = False
    for k in range(cluster - 1, -1, -1):
        # if the amounts is too little ,then remove this cluster
        if len(s[k]) < settings.n_min:
            z = np.delete(z, k, axis=0)
            s.pop(k)
            remove_mark = True
            print('Need removing the cluster number:', k)

    # if not remove, then don't need goto step 2'
    if not remove_mark:
        print('Don\'t need removing')
    cluster = len(z)

    print('finish')

    return z, s, cluster, remove_mark


def new_means(settings, cluster, z, s):
    # 4.
    # move each cluster center to the centroid of the associated set of points
    print('---Step 4: calculate new means z')

    for k in range(cluster):
        # sum_x is the sum of [r, g, b] value of each set
        sum_x = np.zeros([3])
        for i in range(len(s[k])):
            sum_x += settings.data[s[k][i][0], s[k][i][1], :]
        z[k] = (sum_x / (len(s[k])))

    print('finish')

    return z


def average_distance(settings, cluster, z, s):
    # 5.
    # calculate the average distance of points of s[i]
    print('---Step 5: calculate delta')

    delta = []
    delta_sum = 0

    for k in range(cluster):
        sum_dist = 0
        for i in range(len(s[k])):
            sum_dist += np.linalg.norm(settings.data[s[k][i][0], s[k][i][1], :] - z[k, :])
        delta.append(sum_dist / len(s[k]))
        delta_sum += (delta[k] * len(s[k]))

    print('finish')

    return delta_sum / settings.pixels, delta


def goto_merge(settings, cluster, iter_num):
    # 6.
    # estimate if goto merge step
    print('---Step 6: go to merge?')

    if iter_num == settings.i_max:
        settings.L_min = 0
        print('Yes and it\'s the Last')

        return True

    if (2 * cluster > settings.k_init) and ((iter_num % 2 == 0) or (2 * cluster >= 4 * settings.k_init)):
        print('Yes')

        return True

    print('No')

    return False


def vector(settings, cluster, z, s):
    # 7.
    # compute the vector v[k]
    print('---Step 7: calculate vectors')

    v = np.empty([cluster, 3])
    v_max = np.empty([cluster])
    j_max = np.empty([cluster])
    for k in range(cluster):
        for d in range(3):
            sum_vector = 0
            for i in range(len(s[k])):
                sum_vector += pow((settings.data[s[k][i][0], s[k][i][1], d] - z[k][d]), 2)
            v[k][d] = math.sqrt(sum_vector / (len(s[k])))

    # find the max coordinate
    for k in range(cluster):
        v_max[k] = np.amax(v[k, :])
        j_max[k] = ((np.where(v[k, :] == np.amax(v[k, :])))[0].tolist())[0]

    print('finish')

    return v_max, j_max


def cluster_split(settings, cluster, z, s, v_max, j_max, delta, delta_sum):
    # 8.
    # calculate the standard deviation and if split
    print('---Step 8: Split k...')

    split = False
    for k in range(cluster):
        if v_max[k] > settings.sigma_max and (
                ((delta[k] > delta_sum) and (len(s[k]) > (2 * (settings.n_min + 1)))) or (2 * k <= settings.k_init)):
            split = True
            print('cluster is split, need new Sets')
            for d in range(3):
                add = np.copy(z[k])
                if d == j_max[k]:
                    z[k][d] += v_max[k]
                    add[d] -= v_max[k]
                    add = add[np.newaxis, :]
                    z = np.r_[z, add]
    cluster = len(z)

    if split:
        print('Need split')
    else:
        print('Don\'t need split')

    print('finish')

    return z, split, cluster


def pairs_dist(z, cluster):
    # 9.
    # compute the pairwise inter_cluster distances between all distinct pairs of cluster centers
    print('---Step 9: calculating distance of pairs')

    dist_clu = np.empty([cluster, cluster])
    for i in range(cluster):
        for j in range(cluster):
            dist_clu[i][j] = np.linalg.norm(z[i] - z[j])

    print('finish')

    return dist_clu


def cluster_merge(settings, cluster, z, s, dist_clu):
    # 10.
    print('---Step 10: Merging...')

    dist_clu_oder = []
    merge_index = []

    for i in range(cluster):
        for j in range(cluster):
            if i < j:
                dist_clu_oder.append([dist_clu[i][j], i, j])

    dist_clu_oder = np.array(dist_clu_oder)

    # only 1 cluster don't need to merge
    if cluster <= 1:
        return z, s, 1
    else:
        index = np.lexsort([dist_clu_oder[:, 0]])
        dist_clu_oder = dist_clu_oder[index, :]

        for i in range(settings.P_max):
            if dist_clu_oder[i][0] < settings.L_min:
                index_i = int(dist_clu_oder[i][1])
                index_j = int(dist_clu_oder[i][2])
                s[i] = s[i] + s[j]
                z[index_i] = (len(s[index_i]) * z[index_i] + len(s[index_j]) * z[index_j]) / (
                        len(s[index_i]) + len([s[index_j]]))
                merge_index.append(int(dist_clu_oder[i][2]))

        # check if there are cluster pair need to merge
        if len(merge_index) != 0:
            print(len(merge_index), 'clusters need merge')
            z = np.delete(z, merge_index, axis=0)
            s = [s[i] for i in range(0, len(s), 1) if i not in merge_index]
        else:
            print('Don\'t need merge')

        print('finish')

        return z, s, len(s)


def draw(settings, cluster, z, s):
    new_images = np.empty([settings.width, settings.height, 3])  # (500, 500, 3)

    new_images_color = np.copy(new_images)
    rand_color = np.copy(z)
    for k in range(cluster):

        # random color
        rand_color[k] = np.random.rand(1, 3)
        for i in range(len(s[k])):
            new_images[s[k][i][0], s[k][i][1], :] = z[k]  # (cluster, 3)
            new_images_color[s[k][i][0], s[k][i][1], :] = rand_color[k]

    plt.subplot(1, 3, 1)
    plt.imshow(new_images)

    plt.subplot(1, 3, 2)
    plt.imshow(new_images_color)

    plt.subplot(1, 3, 3)
    plt.imshow(settings.data)

    plt.show()


def main():
    time_start = time.time()
    settings = Settings()
    cluster = settings.k_init
    iter_num = 0
    cluster_list = [cluster]

    # 1.
    z = clu_center_select(settings, cluster)

    while iter_num < settings.i_max:

        split_mark = True
        while split_mark and iter_num < settings.i_max:
            split_mark = False

            remove_mark = True
            while remove_mark and iter_num < settings.i_max:
                remove_mark = False
                iter_num += 1
                cluster_list.append(cluster)
                print('=================', iter_num, '=================')
                print('Existing cluster number: ', cluster)

                # 2. get the cluster Set of pixel coordinates
                s = assign_points(settings, cluster, z)

                # 3. remove the too few cluster
                z, s, cluster, remove_mark = remove_cluster(settings, cluster, z, s)

                # 4.
                z = new_means(settings, cluster, z, s)
            # 5.
            delta_sum, delta = average_distance(settings, cluster, z, s)

            # 6.
            if goto_merge(settings, cluster, iter_num):
                break
            else:
                v_max, j_max = vector(settings, cluster, z, s)
                z, split_mark, cluster = cluster_split(settings, cluster, z, s, v_max, j_max, delta, delta_sum)
        # 9.
        dist_clu = pairs_dist(z, cluster)

        # 10.
        z, s, cluster = cluster_merge(settings, cluster, z, s, dist_clu)

    time_end = time.time()

    print('=================End=================')
    print('clusters:', cluster, 'iterations:', iter_num, 'average time per iteration: ',
          (time_end - time_start) / iter_num)
    print('cluster list of all process: ', cluster_list)

    draw(settings, cluster, z, s)


if __name__ == '__main__':
    main()
