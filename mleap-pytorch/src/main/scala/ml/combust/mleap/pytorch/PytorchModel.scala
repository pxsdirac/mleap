package ml.combust.mleap.pytorch

import scala.util.control.NoStackTrace

import ml.combust.mleap.core.Model
import ml.combust.mleap.core.types.{ScalarType, StructType, TensorType}
import ml.combust.mleap.pytorch.PytorchModel.{DoubleType, FloatType, InputElementType}
import ml.combust.mleap.pytorch.converter.MleapConverter
import ml.combust.mleap.tensor.Tensor
import org.pytorch.{IValue, Module}

object PytorchModel{
  sealed trait InputElementType
  case object DoubleType extends InputElementType
  case object FloatType extends InputElementType

}

/**
  *
  * @param module
  * @param rawBytes raw bytes of pytorch model, there is no save api for module, so we need the raw bytes to be used in storing the model.
  * @param numFeatures
  */
case class PytorchModel(module: Module,
                        rawBytes:Array[Byte],
                        numFeatures: Int,
                        inputElementType: InputElementType
                       ) extends Model {
  override def inputSchema: StructType = StructType("features" -> TensorType.Double(numFeatures)).get

  override def outputSchema: StructType = StructType("prediction" -> ScalarType.Double.nonNullable).get

  val extractOutput:IValue => Seq[Double] = inputElementType match {
    case DoubleType => (iValue:IValue) => iValue.toTensor.getDataAsDoubleArray.toSeq
    case FloatType => (iValue:IValue) => iValue.toTensor.getDataAsFloatArray.toSeq.map(_.toDouble)
  }
  def predict(tensor:Tensor[Double]):Seq[Double] = {
    val pytorchTensor = MleapConverter.convert(tensor,inputElementType)
    val shape = pytorchTensor.shape()
    require(shape.last.toInt == numFeatures, s"the length of features should be $numFeatures, but actually it's ${shape.last.toInt}")
    val iValue = IValue.from(pytorchTensor)
    val prediction = module.forward(iValue)
    val value = extractOutput(prediction)
    println("******")
    println(shape.mkString(","))
    println(pytorchTensor.getDataAsFloatArray.mkString(","))
    println(value.mkString(","))
    println("******")
    value
  }
}
