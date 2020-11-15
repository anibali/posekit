from tempfile import NamedTemporaryFile

import numpy as np
from numpy.testing import assert_allclose

from posekit.io import Mocap, save_mocap
from posekit.io.c3d_mocap import save_c3d_mocap, load_c3d_mocap
from posekit.io.json_mocap import save_json_mocap, load_json_mocap
from posekit.io.trc_mocap import save_trc_mocap, load_trc_mocap, save_opensim_trc_mocap
from posekit.skeleton import skeleton_registry


def test_c3d_mocap():
    with NamedTemporaryFile(suffix='.c3d') as f:
        joints = np.random.rand(100, 16, 3)
        mocap1 = Mocap(joints, 'mpii_16j', 50)
        save_c3d_mocap(mocap1, f.name)
        mocap2 = load_c3d_mocap(f.name)
    assert mocap2.skeleton_name == mocap1.skeleton_name
    assert mocap2.sample_rate == mocap1.sample_rate
    assert_allclose(mocap2.joint_positions, joints)


def test_json_mocap():
    with NamedTemporaryFile(suffix='.json') as f:
        joints = np.random.rand(10, 16, 3)
        mocap1 = Mocap(joints, 'mpii_16j', 50)
        save_json_mocap(mocap1, f.name)
        mocap2 = load_json_mocap(f.name)
    assert mocap2.skeleton_name == mocap1.skeleton_name
    assert mocap2.sample_rate == mocap1.sample_rate
    assert_allclose(mocap2.joint_positions, joints, rtol=0, atol=1e-6)


def test_trc_mocap():
    with NamedTemporaryFile(suffix='.trc') as f:
        joints = np.random.rand(10, 16, 3)
        mocap1 = Mocap(joints, 'mpii_16j', 50)
        save_trc_mocap(mocap1, f.name)
        mocap2 = load_trc_mocap(f.name)
    assert mocap2.skeleton_name == mocap1.skeleton_name
    assert mocap2.sample_rate == mocap1.sample_rate
    assert_allclose(mocap2.joint_positions, joints, rtol=0, atol=1e-6)


def test_opensim_trc_mocap():
    skeleton = skeleton_registry['mpii_16j']
    with NamedTemporaryFile(suffix='.trc') as f:
        joints = np.random.rand(10, 16, 3)
        mocap1 = Mocap(joints, skeleton.name, 50)
        save_opensim_trc_mocap(mocap1, f.name)
        mocap2 = load_trc_mocap(f.name)
    assert mocap2.skeleton_name == mocap1.skeleton_name
    assert mocap2.sample_rate == mocap1.sample_rate
    joints_opensim = joints.copy()
    joints_opensim -= joints_opensim[0, skeleton.root_joint_id]
    joints_opensim[:, :, [1, 2]] *= -1
    assert_allclose(mocap2.joint_positions, joints_opensim, rtol=0, atol=1e-6)


def test_save_mocap(mocker):
    stub = mocker.patch('posekit.io.save_c3d_mocap')
    joints = np.random.rand(1, 16, 3)
    mocap = Mocap(joints, 'mpii_16j', 50)
    save_mocap(mocap, 'test.c3d')
    assert stub.call_args_list == [mocker.call(mocap, 'test.c3d')]
