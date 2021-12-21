package ml.combust.mleap.pytorch.converter

import scala.reflect.ClassTag

import ml.combust.mleap.pytorch.PytorchModel.{DoubleType, FloatType, InputElementType}
import ml.combust.mleap.tensor.Tensor
import org.pytorch


object MleapConverter {
  def convert[T: ClassTag](value: Tensor[T], inputElementType: InputElementType): pytorch.Tensor = {
    val dense = value.toDense
    val shape = dense.dimensions.map(_.toLong).toArray
    value.base.runtimeClass match {
      case Tensor.ByteClass =>
        val data = dense.values.asInstanceOf[Array[Byte]]
        pytorch.Tensor.fromBlob(data,shape)
      case Tensor.IntClass =>
        val data = dense.values.asInstanceOf[Array[Int]]
        pytorch.Tensor.fromBlob(data,shape)
      case Tensor.LongClass =>
        val data = dense.values.asInstanceOf[Array[Long]]
        pytorch.Tensor.fromBlob(data,shape)
      case Tensor.FloatClass =>
        val data = dense.values.asInstanceOf[Array[Float]]
        inputElementType match {
          case DoubleType => pytorch.Tensor.fromBlob(data.map(_.toDouble),shape)
          case FloatType => pytorch.Tensor.fromBlob(data,shape)
        }
      case Tensor.DoubleClass =>
        val data = dense.values.asInstanceOf[Array[Double]]
        inputElementType match {
          case DoubleType => pytorch.Tensor.fromBlob(data,shape)
          case FloatType => pytorch.Tensor.fromBlob(data.map(_.toFloat),shape)
        }
      case Tensor.StringClass =>
        throw new IllegalArgumentException("string class is not supported in Tensor of pytorch")
      case Tensor.ByteStringClass =>
        throw new IllegalArgumentException("byte string is not supported in Tensor of pytorch")
      case _ =>
        throw new IllegalArgumentException(s"unsupported tensor type ${value.getClass}[${value.base.runtimeClass}]")
    }

  }
}
