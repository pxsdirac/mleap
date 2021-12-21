import java.io.File
import java.nio.file.{Files, Path, Paths}

import scala.util.Random

import ml.combust.mleap.core.types.{BasicType, NodeShape, StructField, StructType, TensorType}
import ml.combust.mleap.pytorch.{PytorchModel, PytorchTransformer}
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import ml.combust.mleap.tensor.DenseTensor
import org.pytorch.Tensor

object Main extends App {
  println(System.getProperty("java.library.path"))
  import org.pytorch.IValue
  import org.pytorch.Module
  import java.util
  val path = getClass.getClassLoader.getResource("scriptmodule.pt").getPath
  val mod = Module.load(path)
  val row1 = (0 until 3).map{_.toDouble}.toArray      // [0.0, 1.0, 2.0]
  val row2 = (0 until 3).map{_.toDouble + 1}.toArray  // [1.0, 2.0, 3.0]
  def sum(data:Array[Double], shape:Array[Long]):Unit = {
    val result = mod.forward(IValue.from( Tensor.fromBlob(data,shape)))
    val sumValue = result.toTensor.getDataAsDoubleArray.mkString("[",",","]")
    println(sumValue)
  }
  sum(row1,Array(1,3))             // [3.0]
  sum(row2, Array(1,3))            // [6.0]
  sum(row1 ++ row2, Array(2,3))    // [3.0,6.0]



  val bytes = Files.readAllBytes(Paths.get(path))
  val pytorchModel = PytorchModel(mod,bytes, 157,PytorchModel.FloatType)
  val shape = NodeShape()
    .withInput("features","features")
    .withOutput("prediction","prediction")
  val transformer = new PytorchTransformer(model = pytorchModel,shape = shape)
  val data1 = (0 until 157).map{_ => Random.nextDouble()}.toArray
  val data2 = (0 until 157).map{_ => Random.nextDouble()}.toArray
  val data3 = (0 until 157).map{_ => Random.nextDouble()}.toArray

  val schema = StructType(StructField("features", TensorType(BasicType.Double,dimensions = Seq(1,157)))).get
  val dataset = Seq(Row(DenseTensor.apply((data1 ++ data3),Seq(2, 157))),
    Row(DenseTensor.apply(data2,Seq(1,157))),
    Row(DenseTensor.apply(data3,Seq(1,157))))
  val frame = DefaultLeapFrame(schema, dataset)
  val frame2 = transformer.transform(frame).get
  println(frame2.collect().map{row => row.getDouble(1)})

  val result = mod.forward(IValue.from(Tensor.fromBlob((data1 ++ data2).map(_.toFloat),Array[Long](2,157))))
  val output = result.toTensor.getDataAsFloatArray.toSeq
  println(output)

}
