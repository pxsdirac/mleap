package ml.combust.mleap.pytorch

import java.nio.file.Files

import scala.util.{Failure, Random, Success, Try}

import ml.combust.bundle.BundleContext
import ml.combust.bundle.dsl.{Model, Value}
import ml.combust.bundle.op.OpModel
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.pytorch.PytorchModel.{DoubleType, FloatType, InputElementType}
import ml.combust.mleap.runtime.MleapContext
import org.pytorch.{IValue, Module, Tensor}

class PytorchTransformerOp extends MleapOp[PytorchTransformer, PytorchModel] {
  /** Type class for the underlying model.
    */
  override val Model: OpModel[MleapContext, PytorchModel] = new OpModel[MleapContext,PytorchModel] {
    /** Class of the model.
      */
    override val klazz: Class[PytorchModel] = classOf[PytorchModel]

    /** Get the name of the model.
      *
      * @return name of the model
      */
    override def opName: String = "pytorch"

    /** Store the model.
      *
      * Store all standard parameters to the model's attribute list.
      * Store all non-standard parameters like a decision tree to files.
      *
      * Attributes saved to the writable model will be serialized for you
      * to JSON or Protobuf depending on the selected [[ml.combust.bundle.serializer.SerializationFormat]].
      *
      * @param model   writable model to store model attributes in
      * @param obj     object to be stored in Bundle.ML
      * @param context bundle context for custom types
      * @return writable model to be serialized
      */
    override def store(model: Model, obj: PytorchModel)(implicit context: BundleContext[MleapContext]): Model = {
      val out = Files.newOutputStream(context.file("pytorch.pt"))
      out.write(obj.rawBytes)
      out.close()

      model.withValue("num_features", Value.int(obj.numFeatures))
    }

    def determineInputElementType(module: Module, numFeatures:Int):Try[InputElementType] = {
      Try {
        val data = (0 until numFeatures).map{_ => Random.nextDouble()}.toArray
        val tensor = Tensor.fromBlob(data,Array(numFeatures.toLong))
        val prediction = module.forward(IValue.from(tensor))
        prediction.toTensor.getDataAsDoubleArray
        DoubleType
      }.orElse {
        Try {
          val data = (0 until numFeatures).map { _ => Random.nextFloat() }.toArray
          val tensor = Tensor.fromBlob(data, Array(numFeatures.toLong))
          val prediction = module.forward(IValue.from(tensor))
          prediction.toTensor.getDataAsFloatArray
          FloatType
        }
      }
    }

    /** Load the model.
      *
      * Load all standard parameters from the model attributes.
      * Load all non-standard parameters like decision trees from the custom files.
      *
      * @param model   model and attributes read from Bundle.ML
      * @param context bundle context for custom types
      * @return reconstructed ML model from the model and context
      */
    override def load(model: Model)(implicit context: BundleContext[MleapContext]): PytorchModel = {
      val path = context.file("pytorch.pt")
      val rawBytes = Files.readAllBytes(path)
      val module = Module.load(path.toFile.getAbsolutePath)
      val featureNumbers = model.value("num_features").getInt
      determineInputElementType(module, featureNumbers) match {
        case Success(inputElementType) => PytorchModel(module,rawBytes,featureNumbers,inputElementType)
        case Failure(exception) => throw exception
      }
    }
  }

  /** Get the underlying model of the node.
    *
    * @param node node object
    * @return underlying model object
    */
  override def model(node: PytorchTransformer): PytorchModel = node.model
}
