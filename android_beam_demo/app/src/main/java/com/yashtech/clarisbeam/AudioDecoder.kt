package com.yashtech.clarisbeam

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import java.nio.ByteBuffer

object AudioDecoder {
    private const val TAG = "AudioDecoder"

    fun decodeAssetToPcm(context: Context, assetName: String): Pair<FloatArray, Int>? {
        val extractor = MediaExtractor()
        try {
            val afd = context.assets.openFd(assetName)
            extractor.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
            afd.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error setting data source: ${e.message}")
            return null
        }

        var trackIndex = -1
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                trackIndex = i
                break
            }
        }

        if (trackIndex < 0) return null
        extractor.selectTrack(trackIndex)

        val format = extractor.getTrackFormat(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
        val codec = MediaCodec.createDecoderByType(mime)
        
        // We want 16kHz mono for our Mel extractor
        // Note: Real-time resampling might be needed if the source is not 16kHz
        // For simplicity, we assume the sample is 16kHz or we'll let the user know
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        Log.i(TAG, "Source: $sampleRate Hz, $channels channels")

        codec.configure(format, null, null, 0)
        codec.start()

        val pcmData = mutableListOf<Float>()
        val info = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawOutputEOS = false

        while (!sawOutputEOS) {
            if (!sawInputEOS) {
                val inputBufferIndex = codec.dequeueInputBuffer(10000)
                if (inputBufferIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputBufferIndex)!!
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        codec.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outputBufferIndex = codec.dequeueOutputBuffer(info, 10000)
            if (outputBufferIndex >= 0) {
                val outputBuffer = codec.getOutputBuffer(outputBufferIndex)!!
                val chunk = ShortArray(info.size / 2)
                outputBuffer.asShortBuffer().get(chunk)
                
                // Convert to float and mix to mono if needed
                if (channels == 2) {
                    for (i in chunk.indices step 2) {
                        val left = chunk[i].toFloat() / 32768.0f
                        val right = chunk[i + 1].toFloat() / 32768.0f
                        pcmData.add((left + right) / 2.0f)
                    }
                } else {
                    for (s in chunk) {
                        pcmData.add(s.toFloat() / 32768.0f)
                    }
                }
                
                codec.releaseOutputBuffer(outputBufferIndex, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    sawOutputEOS = true
                }
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        return Pair(pcmData.toFloatArray(), sampleRate)
    }

    fun savePcmToWav(pcmData: FloatArray, outPath: String, sampleRate: Int = 16000) {
        val outStream = java.io.FileOutputStream(outPath)
        val dataSize = pcmData.size * 2
        val fileSize = dataSize + 36

        val header = ByteArray(44)
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (fileSize and 0xff).toByte()
        header[5] = ((fileSize shr 8) and 0xff).toByte()
        header[6] = ((fileSize shr 16) and 0xff).toByte()
        header[7] = ((fileSize shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1
        header[21] = 0
        header[22] = 1 // Mono
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        val byteRate = sampleRate * 2
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = 2 // block align
        header[33] = 0
        header[34] = 16 // bits per sample
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (dataSize and 0xff).toByte()
        header[41] = ((dataSize shr 8) and 0xff).toByte()
        header[42] = ((dataSize shr 16) and 0xff).toByte()
        header[43] = ((dataSize shr 24) and 0xff).toByte()

        outStream.write(header, 0, 44)

        // Convert float to 16-bit PCM
        val audioBytes = ByteArray(dataSize)
        for (i in pcmData.indices) {
            var s = (pcmData[i] * 32768.0f).toInt()
            if (s > 32767) s = 32767
            if (s < -32768) s = -32768
            audioBytes[i * 2] = (s and 0xff).toByte()
            audioBytes[i * 2 + 1] = ((s shr 8) and 0xff).toByte()
        }
        
        outStream.write(audioBytes)
        outStream.close()
    }

    fun playWav(context: Context, file: java.io.File, onCompletion: (() -> Unit)? = null) {
        val mediaPlayer = android.media.MediaPlayer()
        try {
            mediaPlayer.setDataSource(file.absolutePath)
            mediaPlayer.prepare()
            mediaPlayer.start()
            mediaPlayer.setOnCompletionListener {
                it.release()
                onCompletion?.invoke()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to play audio: ${e.message}")
            onCompletion?.invoke()
        }
    }

    fun playAsset(context: Context, assetName: String, onCompletion: (() -> Unit)? = null) {
        val mediaPlayer = android.media.MediaPlayer()
        try {
            val afd = context.assets.openFd(assetName)
            mediaPlayer.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
            afd.close()
            mediaPlayer.prepare()
            mediaPlayer.start()
            mediaPlayer.setOnCompletionListener {
                it.release()
                onCompletion?.invoke()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to play asset: ${e.message}")
            onCompletion?.invoke()
        }
    }
}
