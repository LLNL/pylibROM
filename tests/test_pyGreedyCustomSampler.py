#!/usr/bin/env python
import pytest
import numpy as np
import sys
try:
    # import pip-installed package
    import pylibROM.linalg as linalg
    import pylibROM.algo.greedy as greedy
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM.linalg as linalg
    import _pylibROM.algo.greedy as greedy

from mpi4py import MPI


def test_greedy_custom_sampler_centroid():
    paramPoints = [1.0, 3.0, 6.0]
    caromGreedySampler = greedy.GreedyCustomSampler(paramPoints, False, 0.1, 1, 1, 2, 3, "", "", True, 1, True)
    nextPointToSample = caromGreedySampler.getNextParameterPoint()

    assert nextPointToSample.dim() == 1
    assert nextPointToSample.item(0) == 3.0

    caromGreedySampler.getNextPointRequiringRelativeError()
    caromGreedySampler.setPointRelativeError(100.0)

    localPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    assert localPoint.point is not None
    assert localPoint.point.dim() == 1
    assert localPoint.point.item(0) == 3.0
    assert localPoint.localROM is not None

    caromGreedySampler.setPointErrorIndicator(1.0, 1)
    firstPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    assert firstPoint.point is not None
    assert firstPoint.point.dim() == 1
    assert firstPoint.point.item(0) == 1.0

    caromGreedySampler.setPointErrorIndicator(100.0, 1)
    secondPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    assert secondPoint.point is not None
    assert secondPoint.point.dim() == 1
    assert secondPoint.point.item(0) == 6.0

    caromGreedySampler.setPointErrorIndicator(50.0, 1)
    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    assert nextPointToSample.dim() == 1
    assert nextPointToSample.item(0) == firstPoint.point.item(0)

    caromGreedySampler.getNextPointRequiringRelativeError()
    caromGreedySampler.setPointRelativeError(100.0)

    firstPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    assert nextPointToSample.dim() == 1
    assert nextPointToSample.item(0) == paramPoints[2]

    caromGreedySampler.getNextPointRequiringRelativeError()
    caromGreedySampler.setPointRelativeError(100.0)

    firstPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()


def test_greedy_custom_sampler_multi_dim_centroid():
    item0 = linalg.Vector(2, False)
    item0[0] = 1.0
    item0[1] = 11.0

    item1 = linalg.Vector(2, False)
    item1[0] = 3.0
    item1[1] = 13.0

    item2 = linalg.Vector(2, False)
    item2[0] = 6.0
    item2[1] = 16.0

    paramPoints = [item0, item1, item2]

    caromGreedySampler = greedy.GreedyCustomSampler(paramPoints, False, 0.1, 1, 1, 2, 3, "", "", True, 1, True)

    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    assert nextPointToSample.dim() == 2
    assert nextPointToSample.item(0) == 3.0
    assert nextPointToSample.item(1) == 13.0

    caromGreedySampler.getNextPointRequiringRelativeError()
    caromGreedySampler.setPointRelativeError(100.0)

    localPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()

    assert localPoint.point is not None
    assert localPoint.point.dim() == 2
    assert localPoint.point.item(0) == 3.0
    assert localPoint.point.item(1) == 13.0
    assert localPoint.localROM is not None

    caromGreedySampler.setPointErrorIndicator(1.0, 1)

    firstPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    assert firstPoint.point is not None
    assert firstPoint.point.dim() == 2
    assert firstPoint.point.item(0) == 1.0
    assert firstPoint.point.item(1) == 11.0

    caromGreedySampler.setPointErrorIndicator(100.0, 1)
    secondPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    assert secondPoint.point is not None
    assert secondPoint.point.dim() == 2
    assert secondPoint.point.item(0) == 6.0
    assert secondPoint.point.item(1) == 16.0

    caromGreedySampler.setPointErrorIndicator(50.0, 1)
    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    assert nextPointToSample.dim() == 2
    assert nextPointToSample.item(0) == firstPoint.point.item(0)
    assert nextPointToSample.item(1) == firstPoint.point.item(1)

    caromGreedySampler.getNextPointRequiringRelativeError()
    caromGreedySampler.setPointRelativeError(100.0)

    firstPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()
    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    assert nextPointToSample.dim() == 2
    assert nextPointToSample.item(0) == item2.item(0)
    assert nextPointToSample.item(1) == item2.item(1)

    tmp = caromGreedySampler.getNextPointRequiringRelativeError()
    assert tmp.point is None
    assert tmp.localROM is None

    caromGreedySampler.setPointRelativeError(100.0)
    firstPoint = caromGreedySampler.getNextPointRequiringErrorIndicator()


def test_greedy_save_and_load():
    paramPoints = [1.0, 2.0, 3.0, 99.0, 100., 101.0]

    caromGreedySampler = greedy.GreedyCustomSampler(paramPoints, False, 0.1, 1, 1, 3, 4, "", "", False, 1, True)
    caromGreedySampler.save("greedy_test")

    caromGreedySamplerLoad = greedy.GreedyCustomSampler("greedy_test")
    caromGreedySamplerLoad.save("greedy_test_LOAD")

    pointToFindNearestROM = linalg.Vector(1, False)
    pointToFindNearestROM[0] = 1.0

    closestROM = caromGreedySampler.getNearestROM(pointToFindNearestROM)
    closestROMLoad = caromGreedySamplerLoad.getNearestROM(pointToFindNearestROM)

    # there were no points sampled, so closestROM should be None
    assert closestROM is None
    assert closestROM == closestROMLoad

    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    nextPointToSampleLoad = caromGreedySamplerLoad.getNextParameterPoint()

    assert nextPointToSample.dim() == nextPointToSampleLoad.dim()
    assert nextPointToSample.item(0) == nextPointToSampleLoad.item(0)


def test_greedy_save_and_load_with_sample():
    paramPoints = [1.0, 2.0, 3.0, 99.0, 100., 101.0]

    caromGreedySampler = greedy.GreedyCustomSampler(paramPoints, False, 0.1, 1, 1, 3, 4, "", "", False, 1, True)

    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    assert nextPointToSample.dim() == 1
    assert nextPointToSample.item(0) == 3.0

    # save after sampling a point to test if sampled points are restored
    caromGreedySampler.save("greedy_test")

    caromGreedySamplerLoad = greedy.GreedyCustomSampler("greedy_test")
    caromGreedySamplerLoad.save("greedy_test_LOAD")

    pointToFindNearestROM = linalg.Vector(1, False)
    pointToFindNearestROM[0] = 1.0

    closestROM = caromGreedySampler.getNearestROM(pointToFindNearestROM)
    closestROMLoad = caromGreedySamplerLoad.getNearestROM(pointToFindNearestROM)

    assert closestROM is not None
    assert closestROM.dim() == 1
    assert closestROM.dim() == closestROMLoad.dim()
    assert closestROM.item(0) == 3.0
    assert closestROM.item(0) == closestROMLoad.item(0)

    nextPointToSample = caromGreedySampler.getNextParameterPoint()
    nextPointToSampleLoad = caromGreedySamplerLoad.getNextParameterPoint()

    assert nextPointToSample.dim() == nextPointToSampleLoad.dim()
    assert nextPointToSample.item(0) == nextPointToSampleLoad.item(0)

if __name__ == '__main__':
    pytest.main()
