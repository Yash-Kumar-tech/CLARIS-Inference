package com.yashtech.clarisbeam.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import kotlin.math.max
import kotlin.math.min

data class BeamNodeData(
    val token: Int,
    val score: Float,
    val parentIndex: Int = -1,
    val isCompleted: Boolean = false
)

data class BeamStepData(
    val step: Int,
    val nodes: List<BeamNodeData>
)

/** Decoder ids 0..3 are specials (2 = BOS/EOS); unit u is stored as u + 4. */
fun tokenLabel(token: Int): String = when {
    token == 2 -> "</s>"
    token < 4 -> "·"
    else -> (token - 4).toString()
}

@Composable
fun BeamTreeCanvas(
    steps: List<BeamStepData>,
    modifier: Modifier = Modifier
) {
    if (steps.isEmpty()) return

    var scale by remember { mutableFloatStateOf(1f) }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }

    val textMeasurer = rememberTextMeasurer()
    val primaryColor = MaterialTheme.colorScheme.primary
    val secondaryColor = MaterialTheme.colorScheme.secondary
    val errorColor = MaterialTheme.colorScheme.error
    val onSurfaceColor = MaterialTheme.colorScheme.onSurface

    Canvas(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f))
            .pointerInput(Unit) {
                detectTransformGestures { _, pan, zoom, _ ->
                    scale = max(0.5f, min(3f, scale * zoom))
                    offsetX += pan.x
                    offsetY += pan.y
                }
            }
    ) {
        val columnWidth = 150f * scale
        val rowHeight = 80f * scale
        val nodeRadius = 24f * scale

        val startX = (size.width / 2f) - ((steps.size * columnWidth) / 2f) + offsetX
        val startY = size.height / 2f + offsetY

        // Map step -> (nodeIndex -> Offset)
        val nodePositions = mutableMapOf<Int, MutableMap<Int, Offset>>()

        for ((stepIndex, stepData) in steps.withIndex()) {
            val cx = startX + stepIndex * columnWidth
            val map = mutableMapOf<Int, Offset>()

            val totalNodes = stepData.nodes.size
            val startOffsetY = startY - ((totalNodes - 1) * rowHeight) / 2f

            for ((nodeIndex, node) in stepData.nodes.withIndex()) {
                val cy = startOffsetY + nodeIndex * rowHeight
                val pos = Offset(cx, cy)
                map[nodeIndex] = pos

                // Draw edge to parent
                if (stepIndex > 0 && node.parentIndex >= 0) {
                    val prevStepMap = nodePositions[stepIndex - 1]
                    val parentPos = prevStepMap?.get(node.parentIndex)
                    if (parentPos != null) {
                        val path = Path().apply {
                            moveTo(parentPos.x, parentPos.y)
                            cubicTo(
                                parentPos.x + columnWidth / 2f, parentPos.y,
                                pos.x - columnWidth / 2f, pos.y,
                                pos.x, pos.y
                            )
                        }
                        drawPath(
                            path = path,
                            color = onSurfaceColor.copy(alpha = 0.3f),
                            style = Stroke(width = 3f * scale)
                        )
                    }
                }
            }
            nodePositions[stepIndex] = map
        }

        // Draw nodes after edges so they appear on top
        for ((stepIndex, stepData) in steps.withIndex()) {
            val map = nodePositions[stepIndex] ?: continue
            for ((nodeIndex, node) in stepData.nodes.withIndex()) {
                val pos = map[nodeIndex] ?: continue
                
                val nodeColor = if (node.isCompleted) errorColor else if (stepIndex == steps.size - 1) primaryColor else secondaryColor
                
                drawCircle(
                    color = nodeColor,
                    radius = nodeRadius,
                    center = pos
                )
                
                val tokenResult = textMeasurer.measure(
                    text = tokenLabel(node.token),
                    style = TextStyle(
                        color = Color.White,
                        fontSize = (12f * scale).sp,
                        fontWeight = FontWeight.Bold
                    )
                )
                drawText(
                    textLayoutResult = tokenResult,
                    topLeft = Offset(pos.x - (10f * scale), pos.y - (8f * scale))
                )
                
                val scoreResult = textMeasurer.measure(
                    text = String.format("%.2f", node.score),
                    style = TextStyle(
                        color = onSurfaceColor.copy(alpha = 0.8f),
                        fontSize = (10f * scale).sp
                    )
                )
                drawText(
                    textLayoutResult = scoreResult,
                    topLeft = Offset(pos.x - (14f * scale), pos.y + nodeRadius + (4f * scale))
                )
            }
        }
    }
}
