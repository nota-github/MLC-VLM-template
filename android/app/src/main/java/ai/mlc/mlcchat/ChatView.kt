package ai.mlc.mlcchat

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Environment
import android.provider.MediaStore
import android.text.TextUtils
import android.util.Half
import android.util.Log
import android.widget.Toast
import androidx.annotation.HalfFloat
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.capitalize
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat.startActivityForResult
import androidx.navigation.NavController
import kotlinx.coroutines.launch
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.time.Duration
import java.time.Instant
import org.json.JSONArray
import org.json.JSONObject


var benchmark = false
var cnt = 0
var start: Instant = Instant.now()
var end: Instant = Instant.now()
var dataIdx = 0
var dataLength = 0


@ExperimentalMaterial3Api
@Composable
fun ChatView(
    navController: NavController, chatState: AppViewModel.ChatState, activity: Activity, json:JSONArray
) {
    val localFocusManager = LocalFocusManager.current
    (activity as MainActivity).chatState = chatState

    if (dataLength == 0){
        dataLength = json.length()
    }

    if (benchmark && chatState.chatable() && !(activity as MainActivity).has_image) {
        val entity: JSONObject = json[dataIdx] as JSONObject
        val imagePath = "/storage/emulated/0/DCIM/images/" + entity.getString("image_path")
        val inputText = entity.getString("input_text")
        Log.v("ChatView", imagePath)
        Log.v("ChatView", inputText)

        val bitmap = getImage(imagePath)
        if (bitmap != null) {
            val imageData = bitmapToBytes(bitmap)
            (activity as MainActivity).chatState.requestImage(imageData)
            (activity as MainActivity).has_image = true
        } else {
            Log.v("ChatVew", "Image is Null")
        }
    }

    Scaffold(topBar = {
        TopAppBar(
            title = {
                Text(
                    text = "PhiVA-3.9B",
                    color = MaterialTheme.colorScheme.onPrimary
                )
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary),
            navigationIcon = {
                IconButton(
                    onClick = { navController.popBackStack() },
                    enabled = chatState.interruptable()
                ) {
                    Icon(
                        imageVector = Icons.Filled.ArrowBack,
                        contentDescription = "back home page",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            },
            actions = {
                IconButton(
                    onClick = {
                        chatState.requestResetChat()
                        activity.has_image = false },
                    enabled = chatState.interruptable()
                ) {
                    Icon(
                        imageVector = Icons.Filled.Replay,
                        contentDescription = "reset the chat",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            })
    }, modifier = Modifier.pointerInput(Unit) {
        detectTapGestures(onTap = {
            localFocusManager.clearFocus()
        })
    }) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 10.dp)
        ) {
            val lazyColumnListState = rememberLazyListState()
            val coroutineScope = rememberCoroutineScope()
            Text(
                text = chatState.report.value,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .wrapContentHeight()
                    .padding(top = 5.dp)
            )
            Divider(thickness = 1.dp, modifier = Modifier.padding(vertical = 5.dp))
            LazyColumn(
                modifier = Modifier.weight(9f),
                verticalArrangement = Arrangement.spacedBy(5.dp, alignment = Alignment.Bottom),
                state = lazyColumnListState
            ) {
                coroutineScope.launch {
                    lazyColumnListState.animateScrollToItem(chatState.messages.size)
                }
                items(
                    items = chatState.messages,
                    key = { message -> message.id },
                ) { message ->
                    MessageView(messageData = message, activity)
                }
                item {
                    // place holder item for scrolling to the bottom
                }
            }
            Divider(thickness = 1.dp, modifier = Modifier.padding(top = 5.dp))
            SendMessageView(chatState = chatState, activity, (json[dataIdx] as JSONObject).getString("input_text"))
        }
    }
}

fun compressImage(image: Bitmap): Bitmap? {
    val baos = ByteArrayOutputStream()
    image.compress(Bitmap.CompressFormat.JPEG, 100, baos)
    var options = 100
    while (baos.toByteArray().toString().length / 1024 > 100) {
        baos.reset()
        image.compress(
            Bitmap.CompressFormat.JPEG,
            options,
            baos
        )
        options -= 10
    }
    val isBm =
        ByteArrayInputStream(baos.toByteArray())
    return BitmapFactory.decodeStream(isBm, null, null)
}
fun scaleSize(image: Bitmap, newW : Int, newH: Int): Bitmap {
    //if (image.height == image.width)
    //    return image
    val maxDimension = image.height.coerceAtLeast(image.width)
    val squareBitmap = Bitmap.createBitmap(maxDimension, maxDimension, image.config)
    val canvas = Canvas(squareBitmap)
    val paint = Paint()
    paint.color = Color.rgb(127,127,127)
    canvas.drawRect(0f, 0f, maxDimension.toFloat(), maxDimension.toFloat(), paint)
    if (image.height > image.width) {
        canvas.drawBitmap(image, (image.height-image.width)/2f, 0f, null)
    } else {
        canvas.drawBitmap(image, 0f, (image.width-image.height)/2f, null)
    }
    return Bitmap.createScaledBitmap(image, newW, newH, true)
}

fun getImage(srcPath: String?): Bitmap? {
    if (TextUtils.isEmpty(srcPath))
        return null
    val newOpts = BitmapFactory.Options()

    newOpts.inJustDecodeBounds = true

    newOpts.inJustDecodeBounds = false
    val w = newOpts.outWidth
    val h = newOpts.outHeight

    val hh = 224f
    val ww = 224f

    var be = 1
    if (w > h && w > ww) {
        be = (newOpts.outWidth / ww).toInt()
    } else if (w < h && h > hh) {
        be = (newOpts.outHeight / hh).toInt()
    }
    if (be <= 0) be = 1
    newOpts.inSampleSize = be

    val bitmap = BitmapFactory.decodeFile(srcPath, newOpts)

    return scaleSize(bitmap, 224, 224)
}

fun bitmapToBytes(bitmap: Bitmap): FloatArray{
    val width = bitmap.width
    val height = bitmap.height
    val pixels = FloatArray(3 * height * width)

    for (y in 0 until height){
        for (x in 0 until width) {
            val pixelColor = bitmap.getPixel(x, y)

            val redValue = Color.red(pixelColor)
            val greenValue = Color.green(pixelColor)
            val blueValue = Color.blue(pixelColor) //
            pixels[0 + y * width + x] = redValue / 255f - 0.5f
            pixels[1 * height * width + y * width + x] = greenValue / 255f - 0.5f
            pixels[2 * height * width + y * width + x] = blueValue / 255f - 0.5f
        }
    }
    return pixels
}

@Composable
fun MessageView(messageData: MessageData, activity: Activity) {
    val localActivity : MainActivity = activity as MainActivity
    SelectionContainer {
        if (messageData.role == MessageRole.Bot) {
            Row(
                horizontalArrangement = Arrangement.Start,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = messageData.text,
                    textAlign = TextAlign.Left,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier
                        .wrapContentWidth()
                        .background(
                            color = MaterialTheme.colorScheme.secondaryContainer,
                            shape = RoundedCornerShape(5.dp)
                        )
                        .padding(5.dp)
                        .widthIn(max = 300.dp)
                )

            }
        } else {
            Row(
                horizontalArrangement = Arrangement.End,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (messageData.image_path != "") {
                    var bitmap = getImage(messageData.image_path)
//                    val bitmap = getImage("/storage/emulated/0/DCIM/Camera/20240430_155419.jpg")

                    Log.v("get_image", messageData.image_path)
                    if (bitmap != null) {
                        val imageData = bitmapToBytes(bitmap)
//                        Log.v("get_image", image_data.size.toString())
                        val displayBitmap = Bitmap.createScaledBitmap(bitmap, 384, 384, true)
                        Image(
                            displayBitmap.asImageBitmap(),
                            "",
                            modifier = Modifier
                                .wrapContentWidth()
                                .background(
                                    color = MaterialTheme.colorScheme.secondaryContainer,
                                    shape = RoundedCornerShape(5.dp)
                                )
                                .padding(5.dp)
                                .widthIn(max = 300.dp)
                        )
                        if (!localActivity.has_image) {
                            localActivity.chatState.requestImage(imageData)
                        }
                        localActivity.has_image = true
                    }
                } else {
                    Text(
                        text = messageData.text,
                        textAlign = TextAlign.Right,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier
                            .wrapContentWidth()
                            .background(
                                color = MaterialTheme.colorScheme.primaryContainer,
                                shape = RoundedCornerShape(5.dp)
                            )
                            .padding(5.dp)
                            .widthIn(max = 300.dp)
                    )
                }
            }
        }
    }
}


fun saveTextToFile(context: Context, text: String) {
    val state = Environment.getExternalStorageState()
    if (Environment.MEDIA_MOUNTED == state) {
        val file = File(context.getExternalFilesDir(null), "result.txt")
        try {
            val writer = java.io.FileWriter(file, true)
            writer.append(text)
            writer.close()
            Log.d("MainActivity", "Text saved: $text")
        } catch (e: IOException) {
            e.printStackTrace()
        }
    } else {
        Log.e("MainActivity", "External storage is not writable")
    }
}

@ExperimentalMaterial3Api
@Composable
fun SendMessageView(chatState: AppViewModel.ChatState, activity: Activity, inputText: String) {
    val localFocusManager = LocalFocusManager.current
    val localActivity : MainActivity = activity as MainActivity


    if (benchmark){
        val context = LocalContext.current
        if (chatState.chatable() && cnt % 2 == 0 && dataIdx < dataLength && localActivity.has_image){
            if (dataIdx % 5 == 0){
                Toast.makeText(context, "Data Idx: $dataIdx", Toast.LENGTH_SHORT).show()
            }
            start = Instant.now()
//            val inputText = textList[dataIdx]

            chatState.requestGenerate(inputText)
            cnt += 1
            dataIdx += 1
        }
        if (chatState.chatable() && cnt % 2 == 1 && dataIdx <= dataLength) {
            end = Instant.now()
            val elapsedTime = Duration.between(start, end).toMillis()
            val responseText = chatState.messages[1].text

            val resultString = "$dataIdx!@!@!@$elapsedTime!@!@!@$responseText!@#\n\n"
            saveTextToFile(context, resultString)
            cnt += 1
            chatState.requestResetChat()
            localActivity.has_image = false;
        }
    }



    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .height(IntrinsicSize.Max)
            .fillMaxWidth()
            .padding(bottom = 5.dp)
    ) {
        var text by rememberSaveable { mutableStateOf("") }
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text(text = "Input") },
            modifier = Modifier
                .weight(9f),
        )
        IconButton(
            onClick = {
                val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(activity, intent, 1, null)
                Log.v("get_image", "after startActivityForResult" + activity.image_path)
            },
            modifier = Modifier
                .aspectRatio(1f)
                .weight(1f),
            enabled = (chatState.chatable() && !localActivity.has_image)
        ) {
            Icon(
                imageVector = Icons.Filled.AddAPhoto,
                contentDescription = "use camera",
            )
        }
        IconButton(
            onClick = {
                val intent = Intent()
                intent.setType("image/*")
                intent.setAction(Intent.ACTION_GET_CONTENT)
                startActivityForResult(activity, Intent.createChooser(intent, "Select Picture"), 2, null)
                Log.v("get_image", "after startActivityForResult" + activity.image_path)
            },
            modifier = Modifier
                .aspectRatio(1f)
                .weight(1f),
            enabled = (chatState.chatable() && !localActivity.has_image)
        ) {
            Icon(
                imageVector = Icons.Filled.Photo,
                contentDescription = "select image",
            )
        }
        IconButton(
            onClick = {
                localFocusManager.clearFocus()
                chatState.requestGenerate(text)
                text = ""
            },
            modifier = Modifier
                .aspectRatio(1f)
                .weight(1f),
            enabled = (text != "" && chatState.chatable())
        ) {
            Icon(
                imageVector = Icons.Filled.Send,
                contentDescription = "send message",
            )
        }
    }
}
