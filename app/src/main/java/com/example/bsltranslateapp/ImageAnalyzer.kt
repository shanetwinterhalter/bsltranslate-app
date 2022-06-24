package com.example.bsltranslateapp

import android.content.Context
import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.common.collect.ImmutableList
import com.google.mediapipe.formats.proto.ClassificationProto
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.*
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.*


/**
 * Our custom image analysis class.
 *
 * <p>All we need to do is override the function `analyze` with our desired operations. Here,
 * we compute the average luminosity of the image by looking at the Y plane of the YUV frame.
 */
class ImageAnalyzer(context: Context, listener: VideoOutputListener? = null) : ImageAnalysis.Analyzer {
    private val TAG = "IMAGE_ANALYSIS"
    private lateinit var bitmapBuffer: Bitmap

    private val frameRateWindow = 8
    private val frameTimestamps = ArrayDeque<Long>(5)
    private val listeners = ArrayList<VideoOutputListener>().apply { listener?.let { add(it) } }
    private var lastAnalyzedTimestamp = 0L
    private var framesPerSecond: Double = -1.0
    private var noHandFrames = 0
    private val noSignsToDisplay = 6
    private val concurrentPredsRequired = 4
    private val framePredictions = ArrayDeque<Int>(concurrentPredsRequired)
    private val imageDisplayOutput = ArrayDeque<String>(noSignsToDisplay)
    private var imageOutput : String = ""

    /**
      * Mediapipe definition
      */
    private val handsOptions = HandsOptions.builder()
        .setStaticImageMode(false)
        .setMaxNumHands(2)
        .setRunOnGpu(true)
        .build()
    private val hands = Hands(context, handsOptions)
    private val noDims = 3
    private val coordsPerHand = 21
    private val noHands = 2
    private val numberOfCoords = noDims * coordsPerHand * noHands
    private val framesPerSign = 7
    private val vocab = readVocabFromFile(context.assets.open("stream_cnn_3d_vocab.csv"))
    private val normStats = readNormalizationStats(context.assets.open("stream_cnn_3d_norm_stats.csv"))
    private var coordList = FloatArray(framesPerSign*numberOfCoords)
    private val filesFolder = context.filesDir
    private val modelName = "stream_cnn_3d.pt"
    private val modelPath: String = File(filesFolder, modelName).absolutePath
    // Copy the pytorch model out of assets because it needs a unique file path
    init {
        context.assets.open(modelName).use { input ->
            Files.copy(input, File(filesFolder, modelName).toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }
    private val module: Module = Module.load(modelPath)

    /**
     * Analyzes an image to produce a result.
     *
     * <p>The caller is responsible for ensuring this analysis method can be executed quickly
     * enough to prevent stalls in the image acquisition pipeline. Otherwise, newly available
     * images will not be acquired and analyzed.
     *
     * <p>The image passed to this method becomes invalid after this method returns. The caller
     * should not store external references to this image, as these references will become
     * invalid.
     *
     * @param image image being analyzed VERY IMPORTANT: Analyzer method implementation must
     * call image.close() on received images when finished using them. Otherwise, new images
     * may not be received or the camera may stall, depending on back pressure setting.
     *
     */
    override fun analyze(image: ImageProxy) {

        // If there are no listeners attached, we don't need to perform analysis
        if (listeners.isEmpty()) {
            image.close()
            return
        }

        // Keep track of frames analyzed
        val currentTime = System.currentTimeMillis()
        frameTimestamps.push(currentTime)

        // Compute the FPS using a moving average
        while (frameTimestamps.size >= frameRateWindow) frameTimestamps.removeLast()
        val timestampFirst = frameTimestamps.peekFirst() ?: currentTime
        val timestampLast = frameTimestamps.peekLast() ?: currentTime
        framesPerSecond = 1.0 / ((timestampFirst - timestampLast) /
                frameTimestamps.size.coerceAtLeast(1).toDouble()) * 1000.0
        //Log.i(TAG, "Frames per second analyzed: $framesPerSecond")

        // Analysis could take an arbitrarily long amount of time
        // Since we are running in a different thread, it won't stall other use cases

        lastAnalyzedTimestamp = frameTimestamps.first

        // Send image to Mediapipe
        if (!::bitmapBuffer.isInitialized) {
            bitmapBuffer = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        }
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        bitmapBuffer = bitmapBuffer.rotate(image.imageInfo.rotationDegrees.toFloat())
        hands.send(bitmapBuffer, currentTime)
        // Close image - causes app to send next image
        image.close()

        // Wait for results
        hands.setResultListener {
            if (it.multiHandLandmarks().isNotEmpty()) {
                val frameTensor = convertMpToTensor(it)
                val outputTensor = module.forward(IValue.from(frameTensor)).toTensor()
                val scores = outputTensor.dataAsFloatArray
                var maxScore = -Float.MAX_VALUE
                var maxScoreIdx = -1
                scores.forEachIndexed { idx, scoreIt ->
                    if (scoreIt > maxScore) {
                        maxScore = scoreIt
                        maxScoreIdx = idx
                    }
                }
                framePredictions.push(maxScoreIdx)
                if (framePredictions.size > concurrentPredsRequired) {
                    framePredictions.removeLast()
                }
                Log.i(TAG,"Prediction from frame $currentTime is $maxScoreIdx")
            }
        }

        // Set text output

        // Consider it a prediction only if 4 frames in a row have the same prediction
        var latestPrediction = framePredictions.peekFirst()
        framePredictions.forEach {
            if (it != framePredictions.peekFirst()) {
                latestPrediction = null
            }
        }

        // Write the latest prediction to the screen.
        if (latestPrediction != null && latestPrediction != 0) {
            imageOutput = vocab[latestPrediction].toString()
        }


        listeners.forEach { it(imageOutput) }
    }

    private fun convertMpToTensor(handsResult: HandsResult): Tensor {
        val tensorShape: LongArray = longArrayOf(
            1,
            1,
            framesPerSign.toLong(),
            noDims.toLong(),
            (numberOfCoords/noDims).toLong()
        )
        val (leftIdx, rightIdx) = getHandIdx(handsResult.multiHandedness())
        val (leftHandX, leftHandY, leftHandZ) = convertCoordsToList(handsResult, leftIdx)
        val (rightHandX, rightHandY, rightHandZ) = convertCoordsToList(handsResult, rightIdx)
        coordList = coordList.drop(numberOfCoords).toFloatArray() + leftHandX + rightHandX + leftHandY + rightHandY + leftHandZ + rightHandZ
        return Tensor.fromBlob(coordList, tensorShape)
    }

    private fun getHandIdx(handLabels: ImmutableList<ClassificationProto.Classification>): Pair<Int, Int> {
        val noHands = handLabels.size
        var score: Float
        var leftIdx = -1
        var rightIdx = -1
        val leftHandProb : MutableList<Float> = MutableList(noHands) { 0.toFloat() }

        if (noHands == 2) {
            handLabels.forEachIndexed { idx, it ->
                val handLabel = it.label
                score = if (handLabel == "Left") {
                    it.score
                } else {
                    1 - it.score
                }
                leftHandProb[idx] = score
            }
            leftIdx = leftHandProb.indexOf(leftHandProb.maxOf { it })
            rightIdx = leftHandProb.indexOf(leftHandProb.minOf { it })
        } else if (noHands == 1) {
            val handLabel = handLabels[0].label
            if (handLabel == "Left") {
                leftIdx = 0
            } else if (handLabel == "Right") {
                rightIdx = 0
            }
        }
        return Pair(leftIdx, rightIdx)
    }

    private fun convertCoordsToList(handsResult: HandsResult, handIdx: Int) : Triple<FloatArray, FloatArray, FloatArray> {
        val xVals = FloatArray(numberOfCoords / (noHands * noDims))
        val yVals = FloatArray( numberOfCoords / (noHands * noDims))
        val zVals = FloatArray( numberOfCoords / (noHands * noDims))
        if (handIdx != -1) {
            val hand = handsResult.multiHandWorldLandmarks()[handIdx]
            hand.landmarkList.forEachIndexed { idx, it ->
                val coordIdx = idx * noDims
                // These lines also normalize each coordinate
                xVals[idx] = (it.x - normStats[0][coordIdx]) / normStats[1][coordIdx]
                yVals[idx] = (it.y - normStats[0][coordIdx+1]) / normStats[1][coordIdx+1]
                zVals[idx] = (it.z - normStats[0][coordIdx+2]) / normStats[1][coordIdx+2]
            }
        }
        return Triple(xVals, yVals, zVals)
    }

    private fun readVocabFromFile(fileData: InputStream): Map<Int, String> {
        val data = mutableMapOf<Int, String>()

        fileData.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val (key, value) = line.split(",")
                data[key.toInt()] = value
            }
        }
        Log.i(TAG, "Read vocab file, we have ${data.size} signs")
        return data
    }

    private fun readNormalizationStats(fileData: InputStream): Array<List<Float>> {
        val data : Array<List<Float>> = Array(2) { mutableListOf() }
        fileData.bufferedReader().useLines { lines ->
            lines.forEachIndexed { idx, line ->
                val normStat = mutableListOf<Float>()
                line.split(",").forEach {
                    normStat.add(it.toFloat())
                }
                data[idx] = normStat
            }
        }
        return data
    }

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }
}
