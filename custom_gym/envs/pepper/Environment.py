import pybullet as p
import pybullet_data
from qibullet import SimulationManager
from qibullet import PepperVirtual
import time

class BaseEnvironment():
    def __init__(self):
        pass

    def createPepper(self, translation=[0,0,0], quaternion=[0, 0, 0, 1] ):
        """
        This method creates pepper.
        Implement this in each subclass.
        """
        raise NotImplementedError

    #see pybullet documentation
    #returns the id
    def createObjectFromURDF(self, path, basePosition = [0,0,0], baseOrientation = [0,0,0,1], useMaximalCoordinates=0, useFixedBase=0, flags=0, globalScaling=1.0):
        return p.loadURDF(path, basePosition, baseOrientation, useMaximalCoordinates, useFixedBase, flags, globalScaling)


    def createObjectFromOBJ(self, pathVisual, pathVHACD, baseMass, shapeType=p.GEOM_MESH, meshScale = [1,1,1], basePosition = [0,0,0], baseOrientation = [0,0,0,1], rgbaColor=[1,1,1,1]):
        visualObj = p.createVisualShape(shapeType=shapeType, fileName=pathVisual, meshScale=meshScale, rgbaColor=rgbaColor)
        collisionObj = p.createCollisionShape(shapeType=shapeType, fileName=pathVHACD, meshScale=meshScale)
        bodyObj = p.createMultiBody(baseMass=1, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collisionObj, baseVisualShapeIndex=visualObj, basePosition=basePosition,baseOrientation=baseOrientation)
        
        return bodyObj

    def createObjectFromMJCF(self,path):
        return p.loadMJCF(path)

class Environment_DIRECT(BaseEnvironment):

    def __init__(self):
        super().__init__()

    def createPepper(self, translation=[0,0,0], quaternion=[0, 0, 0, 1] ):
        simulation_manager = SimulationManager()
        client = simulation_manager.launchSimulation(gui=False)
        pepper = simulation_manager.spawnPepper(client, translation, quaternion, spawn_ground_plane=True)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setGravity(0, 0, -9.81)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # pepper = PepperVirtual()
        # pepper.loadRobot(translation, quaternion)
        return pepper


class Environment_GUI(BaseEnvironment):

    def __init__(self):
        #physicsClient = p.connect(p.GUI, options="--mp4=movie.mp4")
        #physicsClient = p.connect(p.DIRECT)
        physicsClient = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.setRealTimeSimulation(1)
        # p.setTimeStep(1./60.)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadMJCF("mjcf/ground_plane.xml")
        # self.objectList = ["bottle", "cup", "mug"]
        super().__init__()

    #creates a Pepper robot
    #returns tthe pepper object
    def createPepper(self, translation=[0,0,0], quaternion=[0, 0, 0, 1] ):
        pepper = PepperVirtual()
        pepper.loadRobot(translation=[0,0,0], quaternion=[0, 0, 0, 1])
        return pepper