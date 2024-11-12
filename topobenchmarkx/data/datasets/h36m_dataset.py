"""Dataset class for Human3.6M dataset (http://vision.imar.ro/human3.6m/description.php)."""

import json
import os
import os.path as osp
import shutil
from typing import ClassVar

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobenchmarkx.data.utils import (
    download_file_from_drive,
)


class H36MDataset(InMemoryDataset):
    r"""Dataset class for Human3.6M dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    # Taken from siMLPe paper: https://github.com/dulucas/siMLPe/tree/c92c537e833443aa55554e4f7956838746550187
    # Pre-processed from the original dataset to be .txt files
    # TODO: not sure if this is legal, so find a way to go from official website with license
    URLS: ClassVar = {
        "H36MDataset": "https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view"
    }

    FILE_FORMAT: ClassVar = {
        "H36MDataset": "zip",
    }

    RAW_FILE_NAMES: ClassVar = {}

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.used_joint_indexes = np.array(
            [
                2,
                3,
                4,
                5,
                7,
                8,
                9,
                10,
                12,
                13,
                14,
                15,
                17,
                18,
                19,
                21,
                22,
                25,
                26,
                27,
                29,
                30,
            ]
        ).astype(np.int64)

        super().__init__(
            root,
            force_reload=True,
        )

        out = fs.torch_load(self.processed_paths[0])
        # print(out)
        # print(len(out))
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        print(self.data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(
            self.root,
            self.name,
        )
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        r"""Download the dataset from a URL and saves it to the raw directory.

        Raises:
            FileNotFoundError: If the dataset URL is not found.
        """
        # Step 1: download data from the source
        # self.url = self.URLS[self.name]
        self.file_format = self.FILE_FORMAT[self.name]
        # download_file_from_drive(
        #     file_link=self.url,
        #     path_to_save=self.raw_dir,
        #     dataset_name=self.name,
        #     file_format=self.file_format,
        # )

        # Step 1 isn't working because it is downloading html with the Antivirus Scan Warning instead.
        # Instead manually plop the zip in the right folder.
        # scp H36MDataset.zip abby@bongo.ece.ucsb.edu:/home/abby/code/TopoBenchmark/topobenchmarkx/data/H36MDataset/raw

        # Step 2: extract file
        step2_already_done = False
        folder = self.raw_dir
        if not step2_already_done:
            filename = f"{self.name}.{self.file_format}"
            path = osp.join(folder, filename)
            print(path)
            extract_zip(path, folder)
            os.unlink(path)  # Delete zip file

        # Step 3: move the extracted files to the folder with corresponding name
        # Move files from osp.join(folder, name_download) to folder
        step3_already_done = False
        if not step3_already_done:
            for subject_dir in os.listdir(
                osp.join(folder, "h36m")
            ):  # self.name)):
                for file in os.listdir(osp.join(folder, "h36m", subject_dir)):
                    if file.endswith("ipynb"):
                        continue
                    shutil.move(
                        osp.join(folder, "h36m", subject_dir, file),
                        osp.join(folder, f"{subject_dir}-{file}"),
                    )

            # Delete osp.join(folder, self.name) dir
            shutil.rmtree(osp.join(folder, "h36m"))

    def process(self) -> None:
        r"""Handle the data for the dataset.

        This method loads the Human3.6M dataset,
        loads it into basic graph form,
        and saves this preprocessed data.

        A lot of this is using functions copied from the siMLPe paper.
        """
        # Step 1: extract the data
        h36m_files = self.load_raw_motion_matrices()
        # print(h36m_files)

        # Step 2: define connections
        # this implementation is kinda sketchy but we ignore...
        skl = H36MSkeleton()
        adj_mat = skl.get_bone_adj_mat()
        edges = skl.get_bone_list()

        # print(adj_mat)
        # print(edges)

        # Step 3: turn them into torch_geometric.data Data objects
        motions = []
        for motion_matrix in h36m_files:
            # motion_matrix.shape = [1064, 22, 3]
            #       time, joints, channels

            # need to flatten so we have: large(time), med(joints), small(channels)
            # so if there are 3 times and 4 joints and 2 channels (ignore the brackets)
            #   [
            #       [j0c0t0, j0c1t0]
            #       [j1c0t0, j1c1t0]
            #       [j2c0t0, j2c1t0]
            #       [j3c0t0, j3c1t0]
            #   ]
            #   [
            #       [j0c0t1, j0c1t1]
            #       [j1c0t1, j1c1t1]
            #       [j2c0t1, j2c1t1]
            #       [j3c0t1, j3c1t1]
            #   ]
            #   [
            #       [j0c0t2, j0c1t2]
            #       [j1c0t2, j1c1t2]
            #       [j2c0t2, j2c1t2]
            #       [j3c0t2, j3c1t2]
            #   ]
            # just read this in order!

            # Step 1: Flatten motion_matrix; node for each time, joint, channel combo.
            flat_motion_nodes = torch.reshape(motion_matrix, (-1,))

            # Step 2: Create superset of all possible edge indices.
            # Edges according to skeleton bones.
            # TODO Do we want self-loops?
            t, j, c = motion_matrix.shape
            small_bones = [(n1 * c, n2 * c) for (n1, n2) in edges]
            all_channel_bones = [
                (n1 + cc, n2 + cc)
                for (n1, n2) in small_bones
                for cc in range(c)
            ]
            all_channel_all_time_bones = [
                (n1 + tt * (j * c), n2 + tt * (j * c))
                for (n1, n2) in all_channel_bones
                for tt in range(t)
            ]

            # print("SMOL:", small_bones)
            # print("ALLC", all_channel_bones)
            # print("AJLKDJKFA", all_channel_all_time_bones)
            # break
            # TODO Edges according to time.
            time_edges = []

            # TODO Edges according to channel.
            channel_edges = []

            # TODO Edges last.
            space_channel_edges = []

            edge_index = [
                all_channel_all_time_bones
                + time_edges
                + channel_edges
                + space_channel_edges
            ]

            # Step 3: Create graph Data objects.
            motion_graph = Data(
                x=motion_matrix,
                edge_index=all_channel_all_time_bones,
            )
            motions.append(motion_graph)

        # Step 4: collate the graphs (using InMemoryDataset)
        self.data, self.slices = self.collate(motions)
        self._data_list = None  # Reset cache.

        # Step 5: save processed data
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )

    # def __len__(self):
    #     # if self._file_length is not None:
    #     #     return self._file_length
    #     return len(self._h36m_files)

    def load_raw_motion_matrices(self, train=True):
        r"""Load raw motion data.

        This method loads the Human3.6M dataset.

        A lot of this is using functions copied from the siMLPe paper.

        Parameters
        ----------
        train : bool
            Probably to delete. Don't understand where train goes.

        Returns
        -------
        np.array
            Raw motion matrices from files TODO WHAT IS THIS.
        """
        # Get train / test split
        if train:
            subj_names = ["S1", "S6", "S7", "S8", "S9"]
        else:
            subj_names = ["S5"]

        file_list = []
        for subj in subj_names:
            for filename in os.listdir(self.raw_dir):
                if filename.startswith(subj):
                    file_list.append(osp.join(self.raw_dir, filename))

        # print(self.raw_dir)
        # print("******", file_list)

        raw_motion_matrices = []
        for path in file_list:
            info = open(path, "r").readlines()

            pose_info = []
            for line in info:
                line = line.strip().split(",")
                if len(line) > 0:
                    pose_info.append(np.array([float(x) for x in line]))
            pose_info = np.array(pose_info)

            T = pose_info.shape[0]
            pose_info = pose_info.reshape(-1, 33, 3)
            pose_info[:, :2] = 0
            pose_info = pose_info[:, 1:, :].reshape(-1, 3)
            pose_info = expmap2rotmat_torch(
                torch.tensor(pose_info).float()
            ).reshape(T, 32, 3, 3)

            xyz_info = rotmat2xyz_torch(pose_info)
            xyz_info = xyz_info[:, self.used_joint_indexes, :]

            raw_motion_matrices.append(xyz_info)

        return raw_motion_matrices

        # TODO deal with this processing step later; might change the reshape / edges

        # def _collect_all(self):
        # Keep align with HisRep dataloader
        self.h36m_seqs = []
        self.data_idx = []
        idx = 0
        for h36m_motion_poses in h36m_files:
            N = len(h36m_motion_poses)
            if (
                N
                < 10  # self.h36m_motion_target_length
                + 50  # self.h36m_motion_input_length
            ):
                continue

            sample_rate = 2
            sampled_index = np.arange(0, N, sample_rate)
            h36m_motion_poses = h36m_motion_poses[sampled_index]

            T = h36m_motion_poses.shape[0]
            h36m_motion_poses = h36m_motion_poses.reshape(T, -1)
            print(h36m_motion_poses.shape)
            self.h36m_seqs.append(h36m_motion_poses)
            valid_frames = np.arange(
                0,
                T
                - 50  # self.h36m_motion_input_length
                - 10  # self.h36m_motion_target_length
                + 1,
                1,  # self.shift_step,
            )

            self.data_idx.extend(
                zip([idx] * len(valid_frames), valid_frames.tolist())
            )
            idx += 1

        return h36m_files

    # def __getitem__(self, index):
    #     idx, start_frame = self.data_idx[index]
    #     frame_indexes = np.arange(
    #         start_frame,
    #         start_frame
    #         + self.h36m_motion_input_length
    #         + self.h36m_motion_target_length,
    #     )
    #     motion = self.h36m_seqs[idx][frame_indexes]
    #     if self.data_aug:
    #         if torch.rand(1)[0] > 0.5:
    #             idx = [i for i in range(motion.size(0) - 1, -1, -1)]
    #             idx = torch.LongTensor(idx)
    #             motion = motion[idx]

    #     h36m_motion_input = (
    #         motion[: self.h36m_motion_input_length] / 1000
    #     )  # meter
    #     h36m_motion_target = (
    #         motion[self.h36m_motion_input_length :] / 1000
    #     )  # meter

    #     h36m_motion_input = h36m_motion_input.float()
    #     h36m_motion_target = h36m_motion_target.float()
    #     return h36m_motion_input, h36m_motion_target


########################
### HELPER FUNCTIONS ###
########################


class H36MSkeleton:
    r"""Dataset class for Human3.6M Skeleton."""

    def __init__(self):
        """
        H36M skeleton with 22 joints.
        """
        self.NUM_JOINTS = 22

        self.bone_list = self.generate_bone_list()
        self.bone_adj_mat = self.generate_bone_adj_mat()

    def generate_bone_list(self):
        r"""Generate bones in H36M skeleton with 22 joints.

        Returns
        -------
        list[tup[int]]
            Edge list with bone links and self links.
        """
        self_links = [(i, i) for i in range(self.NUM_JOINTS)]
        joint_links = [
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (1, 9),
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (10, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (15, 17),
            (10, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (20, 22),
        ]

        return self_links + [(i - 1, j - 1) for (i, j) in joint_links]

    def generate_bone_adj_mat(self):
        r"""Generate adj matrix for H36M skeleton with 22 joints.

        Returns
        -------
        list[list[int]]
            Adj matrix for bone links and self links.
        """
        skl = np.zeros((self.NUM_JOINTS, self.NUM_JOINTS))
        for i, j in self.bone_list:
            skl[j, i] = 1
            skl[i, j] = 1
        return skl

    def get_bone_list(self):
        r"""Getter fn for bone list.

        Returns
        -------
        list[tup[int]]
            Edge list with bone links and self links.
        """
        return self.bone_list[:]

    def get_bone_adj_mat(self):
        r"""Getter fn for bone adj mat.

        Returns
        -------
        list[list[int]]
            Adj matrix for bone links and self links.
        """
        return self.bone_adj_mat[:]


def _some_variables():
    r"""Silly way to store variables but they did it.

    Borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
    We define some variables that are useful to run the kinematic tree

    Returns
    -------
    parent:
        32-long vector with parent-child relationships in the kinematic tree.
    offset:
        96-long vector with bone lenghts.
    rotInd:
        32-long list with indices into angles.
    expmapInd:
        32-long list with indices into expmap angles.
    """

    parent = (
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                1,
                7,
                8,
                9,
                10,
                1,
                12,
                13,
                14,
                15,
                13,
                17,
                18,
                19,
                20,
                21,
                20,
                23,
                13,
                25,
                26,
                27,
                28,
                29,
                28,
                31,
            ]
        )
        - 1
    )

    offset = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            -132.948591,
            0.000000,
            0.000000,
            0.000000,
            -442.894612,
            0.000000,
            0.000000,
            -454.206447,
            0.000000,
            0.000000,
            0.000000,
            162.767078,
            0.000000,
            0.000000,
            74.999437,
            132.948826,
            0.000000,
            0.000000,
            0.000000,
            -442.894413,
            0.000000,
            0.000000,
            -454.206590,
            0.000000,
            0.000000,
            0.000000,
            162.767426,
            0.000000,
            0.000000,
            74.999948,
            0.000000,
            0.100000,
            0.000000,
            0.000000,
            233.383263,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            121.134938,
            0.000000,
            0.000000,
            115.002227,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.034226,
            0.000000,
            0.000000,
            278.882773,
            0.000000,
            0.000000,
            251.733451,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999627,
            0.000000,
            100.000188,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.031437,
            0.000000,
            0.000000,
            278.892924,
            0.000000,
            0.000000,
            251.728680,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999888,
            0.000000,
            137.499922,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
        ]
    )
    offset = offset.reshape(-1, 3)

    rotInd = [
        [5, 6, 4],
        [8, 9, 7],
        [11, 12, 10],
        [14, 15, 13],
        [17, 18, 16],
        [],
        [20, 21, 19],
        [23, 24, 22],
        [26, 27, 25],
        [29, 30, 28],
        [],
        [32, 33, 31],
        [35, 36, 34],
        [38, 39, 37],
        [41, 42, 40],
        [],
        [44, 45, 43],
        [47, 48, 46],
        [50, 51, 49],
        [53, 54, 52],
        [56, 57, 55],
        [],
        [59, 60, 58],
        [],
        [62, 63, 61],
        [65, 66, 64],
        [68, 69, 67],
        [71, 72, 70],
        [74, 75, 73],
        [],
        [77, 78, 76],
        [],
    ]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    r"""Pytorch version of fkl.

    Convert joint angles to joint locations.
    batch pytorch version of the fkl() method above

    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:

    Parameters
    ----------
    rotmat : idk
        Idk.
    parent : idk
        Idk.
    offset : idk
        Idk.
    rotInd : idk
        Idk.
    expmapInd : idk
        Idk.

    Returns
    -------
    idk
        N*joint_n*3.
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = (
        torch.from_numpy(offset)
        .float()
        .to(rotmat.device)
        .unsqueeze(0)
        .repeat(n, 1, 1)
        .clone()
    )
    R = rotmat.view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(
                R[:, i, :, :], R[:, parent[i], :, :]
            ).clone()
            p3d[:, i, :] = (
                torch.matmul(p3d[0, i, :], R[:, parent[i], :, :])
                + p3d[:, parent[i], :]
            )
    return p3d


def expmap2rotmat_torch(r):
    r"""Convert expmap matrix to rotation.

    batch pytorch version ported from the corresponding method above

    Parameters
    ----------
    r : np.array
        Shape=N*3.

    Returns
    -------
    np.array
        Shape=N*3*3.
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = (
        torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device)
        + torch.mul(
            torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1
        )
        + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)),
            torch.matmul(r1, r1),
        )
    )
    return R


def rotmat2xyz_torch(rotmat):
    r"""Convert expmaps to joint locations.

    Parameters
    ----------
    rotmat : np.array
        Shape=N*32*3*3.

    Returns
    -------
    np.array
        Shape=N*32*3.
    """
    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd = _some_variables()
    xyz = fkl_torch(rotmat, parent, offset, rotInd, expmapInd)
    return xyz


if __name__ == "__main__":
    # A = np.array([[ [000, 100, 200, 1],
    #                 [2, 2, 2, 2]],
    #               [ [11, 12, 13, 14],
    #                 [2, 2, 2, 2]],
    #               [ [1, 1, 1, 1],
    #                 [2, 2, 2, 2]]])

    # print(A)
    # print(A.reshape(4*3*2, order="C"))
    # print(A.shape)
    root_data_dir = "/home/abby/code/TopoBenchmark/topobenchmarkx/data/datasets/graph/motion"
    name = "H36MDataset"
    dataset = H36MDataset(
        root=root_data_dir,
        name=name,  # self.parameters["data_name"],
        parameters=None,
    )  # self.parameters,
    # print(dataset)
