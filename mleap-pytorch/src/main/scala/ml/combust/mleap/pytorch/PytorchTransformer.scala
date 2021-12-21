package ml.combust.mleap.pytorch

import scala.util.Try

import ml.combust.mleap.core.types.NodeShape
import ml.combust.mleap.runtime.frame.{FrameBuilder, Transformer}
import ml.combust.mleap.runtime.function.{FieldSelector, UserDefinedFunction}
import ml.combust.mleap.tensor.Tensor


class PytorchTransformer (override val uid: String = Transformer.uniqueName("tensorflow"),
                          override val shape: NodeShape,
                          override val model: PytorchModel) extends Transformer {

  val selector = FieldSelector("features")
  /** Transform a builder using this MLeap transformer.
    *
    * @param builder builder to transform
    * @tparam FB underlying class of builder
    * @return try new builder with transformation applied
    */
  override def transform[FB <: FrameBuilder[FB]](builder: FB): Try[FB] = {
    builder.withColumn("prediction",selector){
      UserDefinedFunction(
        f = (features:Tensor[Double]) => {
          val value = model.predict(features).head
          value
        },
        outputSchema,
        inputSchema
      )
    }
  }
}
