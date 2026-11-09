package com.yashtech.clarisbeam

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yashtech.clarisbeam.ui.BeamNodeData
import com.yashtech.clarisbeam.ui.BeamStepData
import com.yashtech.clarisbeam.ui.BeamTreeCanvas
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

/** One bundled wTIMIT recording and its reference sentence. */
data class Sample(val file: String, val text: String) {
    val asset get() = "samples/$file"
}

/** Beam width of the exported decoder: its batch dimension is fixed at 3 (backend/s2u_inference/export_litert_lm.py). */
const val BEAM_SIZE = 3

data class RunStats(
    val initMs: Double,
    val melMs: Double,
    val encoderMs: Double,
    val prefillMs: Double,
    val decodeMs: Double,
    val vocoderMs: Long,
    val numUnits: Int,
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = lightColorScheme(primary = Color(0xFF3F51B5))) {
                Surface(Modifier.fillMaxSize()) { BeamDemoScreen() }
            }
        }
    }
}

private fun loadSamples(context: Context): List<Sample> =
    context.assets.open("samples/samples.tsv").bufferedReader().readLines()
        .mapNotNull { line -> line.split('\t').takeIf { it.size >= 2 }?.let { Sample(it[0].trim(), it[1].trim()) } }
        .sortedBy { it.file }

/** Reads the search tree the native bridge writes: one JSON object per decoding step. */
private fun parseTrajectory(file: File, beamSize: Int): List<BeamStepData> {
    if (!file.exists()) return emptyList()
    val steps = mutableListOf<BeamStepData>()
    for (line in file.readLines()) {
        if (!line.startsWith("{")) continue
        runCatching {
            val obj = JSONObject(line)
            val arr = obj.getJSONArray("nodes")
            val nodes = (0 until minOf(beamSize, arr.length())).map {
                val n = arr.getJSONObject(it)
                BeamNodeData(n.getInt("t"), n.getDouble("s").toFloat(), n.getInt("p"), n.optBoolean("c", false))
            }
            steps.add(BeamStepData(obj.getInt("step"), nodes))
        }
    }
    return steps
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BeamDemoScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val samples = remember { loadSamples(context) }
    var selected by remember { mutableStateOf(samples.firstOrNull()) }
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("Pick a sample and run beam search.") }
    var units by remember { mutableStateOf<List<Int>>(emptyList()) }
    var steps by remember { mutableStateOf<List<BeamStepData>>(emptyList()) }
    var stats by remember { mutableStateOf<RunStats?>(null) }
    var synthPath by remember { mutableStateOf<String?>(null) }

    fun run(sample: Sample) {
        busy = true
        units = emptyList()
        steps = emptyList()
        stats = null
        synthPath = null
        status = "Decoding audio…"
        scope.launch(Dispatchers.Default) {
            try {
                val cacheDir = context.cacheDir
                val (wave, rate) = AudioDecoder.decodeAssetToPcm(context, sample.asset)
                    ?: error("cannot decode ${sample.file}")
                File(cacheDir, "beam_trajectory.txt").delete()

                withContext(Dispatchers.Main) { status = "Running encoder and beam search (width $BEAM_SIZE)…" }
                val json = JSONObject(
                    NativeInference.runBeamSearch(context.assets, cacheDir.absolutePath, wave, rate, BEAM_SIZE)
                )
                if (!json.getBoolean("ok")) error(json.getString("error"))
                val predicted = NativeInference.getLastUnits().toList()
                val tree = parseTrajectory(File(cacheDir, "beam_trajectory.txt"), BEAM_SIZE)
                withContext(Dispatchers.Main) {
                    units = predicted
                    steps = tree
                    status = "Synthesizing speech…"
                }

                val t0 = System.currentTimeMillis()
                val audio = NativeInference.synthesize(context.assets, cacheDir.absolutePath, predicted.toIntArray())
                val vocoderMs = System.currentTimeMillis() - t0
                var out: File? = null
                if (audio.isNotEmpty()) {
                    out = File(cacheDir, "beam_demo_output.wav")
                    AudioDecoder.savePcmToWav(audio, out.absolutePath, 16000)
                }
                withContext(Dispatchers.Main) {
                    stats = RunStats(
                        json.getDouble("initMs"), json.getDouble("melMs"), json.getDouble("encoderMs"),
                        json.getDouble("prefillMs"), json.getDouble("decodeMs"), vocoderMs, predicted.size,
                    )
                    synthPath = out?.absolutePath
                    status = if (out != null) "Done." else "Beam search finished, but synthesis failed."
                }
                out?.let { AudioDecoder.playWav(context, it) }
            } catch (e: Throwable) {
                withContext(Dispatchers.Main) { status = "❌ ${e.message}" }
            } finally {
                withContext(Dispatchers.Main) { busy = false }
            }
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("CLARIS beam search demo", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text(
            "Whispered wTIMIT speech → units (encoder + beam-search decoder) → speech " +
                "(duration predictor + vocoder), all on device.",
            style = MaterialTheme.typography.bodyMedium,
        )

        var menuOpen by remember { mutableStateOf(false) }
        ExposedDropdownMenuBox(
            expanded = menuOpen,
            onExpandedChange = { if (!busy) menuOpen = it },
        ) {
            OutlinedTextField(
                value = selected?.text ?: "",
                onValueChange = {},
                readOnly = true,
                enabled = !busy,
                label = { Text("wTIMIT sample${selected?.let { " (${it.file})" } ?: ""}") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = menuOpen) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                samples.forEach { s ->
                    DropdownMenuItem(
                        text = {
                            Column {
                                Text(s.text, style = MaterialTheme.typography.bodyMedium)
                                Text(
                                    s.file, style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                                )
                            }
                        },
                        onClick = {
                            selected = s
                            menuOpen = false
                        },
                    )
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(
                onClick = { selected?.let { AudioDecoder.playAsset(context, it.asset) } },
                enabled = !busy && selected != null,
                modifier = Modifier.weight(1f),
            ) { Text("Play input") }
            Button(
                onClick = { selected?.let { run(it) } },
                enabled = !busy && selected != null,
                modifier = Modifier.weight(1f),
            ) { Text(if (busy) "Running…" else "Run beam search") }
        }
        Text("Beam width: $BEAM_SIZE (fixed by the exported decoder)", style = MaterialTheme.typography.labelMedium)

        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (busy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
            Text(status, style = MaterialTheme.typography.bodyMedium)
        }

        if (units.isNotEmpty()) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Predicted units (${units.size})", fontWeight = FontWeight.Bold)
                    Text(units.joinToString(" "), fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                    selected?.let { Text("Reference: ${it.text}", style = MaterialTheme.typography.bodySmall) }
                }
            }
        }

        stats?.let { s ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("Timing", fontWeight = FontWeight.Bold)
                    listOf(
                        "Model load (first run only)" to s.initMs,
                        "Mel features" to s.melMs,
                        "Encoder" to s.encoderMs,
                        "Decoder prefill" to s.prefillMs,
                        "Beam search loop" to s.decodeMs,
                        "Durations + vocoder" to s.vocoderMs.toDouble(),
                    ).forEach { (name, ms) ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(name, style = MaterialTheme.typography.bodySmall)
                            Text("%.0f ms".format(ms), style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                        }
                    }
                    if (s.numUnits > 0) {
                        Text("%.1f ms per decoded unit".format(s.decodeMs / s.numUnits), style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }

        synthPath?.let { path ->
            OutlinedButton(onClick = { AudioDecoder.playWav(context, File(path)) }, modifier = Modifier.fillMaxWidth()) {
                Text("Play synthesized speech")
            }
        }

        if (steps.isNotEmpty()) {
            Text("Beam search tree", style = MaterialTheme.typography.titleMedium)
            Text(
                "Each column is a decoding step. Circles are the $BEAM_SIZE surviving hypotheses (unit id) with their " +
                    "cumulative log-probability; red ones ended with </s>. Drag or pinch to explore.",
                style = MaterialTheme.typography.bodySmall,
            )
            Box(
                Modifier.fillMaxWidth().height(380.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .border(1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f), RoundedCornerShape(12.dp)),
            ) { BeamTreeCanvas(steps) }
        }
        Spacer(Modifier.height(24.dp))
    }
}
