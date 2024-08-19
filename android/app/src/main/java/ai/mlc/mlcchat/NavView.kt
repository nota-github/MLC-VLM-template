package ai.mlc.mlcchat

import android.app.Activity
import android.content.Intent
import android.net.Uri
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.ClickableText
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.Column

@ExperimentalMaterial3Api
@Composable
fun NavView(activity: Activity, appViewModel: AppViewModel = viewModel()) {
    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = "home") {
        composable("home") { StartView(navController, appViewModel) }
        composable("chat") { ChatView(navController, appViewModel.chatState, activity) }
        composable("open_source_list") { OpenSourceListView() }  // 새로운 화면 추가
    }
}


@Composable
fun OpenSourceListView() {
    val context = LocalContext.current
    LazyColumn(modifier = Modifier.padding(16.dp)) {
        item {
            // Title
            Text(
                text = "NOTICE FOR OPEN SOURCE LICENSE",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }

        item {
            // Description
            Text(
                text = "This software is based on the MLC LLM application distributed under the Apache 2.0 license, and has been modified and redistributed by Nota Inc. The original authors are the mlc-ai/mlc-llm project team.",
                fontSize = 13.sp,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }

        item {
            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }

        // License Entries
        val licenses = listOf(
            "AndroidX" to listOf(
                "https://github.com/androidx",
                "Copyright © 2024 The Android Open Source Project.",
                "Apache License 2.0"
            ),
            "JetBrains Kotlin" to listOf(
                "https://github.com/JetBrains/kotlin/blob/master/license/README.md",
                "Copyright 2010-2024 JetBrains s.r.o. and Kotlin Programming Language contributors",
                "Apache License 2.0"
            ),
            "GSON" to listOf(
                "https://github.com/google/gson",
                "Copyright (C) 2008-Google Inc.",
                "Apache License 2.0"
            ),
            "Material Components for Android" to listOf(
                "https://github.com/material-components/material-components-android",
                "Copyright 2019 The Android Open Source Project.",
                "Apache License 2.0"
            ),
            "MLC-LLM" to listOf(
                "https://github.com/mlc-ai/mlc-llm",
                "https://huggingface.co/mlc-ai",
                "Copyright MLC-AI.",
                "Apache License 2.0"
            ),
            "LLaVA" to listOf(
                "https://github.com/haotian-liu/LLaVA",
                "https://huggingface.co/llava-hf/llava-1.5-7b-hf",
                "Apache License 2.0"
            ),
            "Phi3" to listOf(
                "https://github.com/microsoft/Phi-3CookBook",
                "https://huggingface.co/microsoft",
                "Copyright (c) Microsoft Corporation.",
                "MIT License"
            ),
        )

        items(licenses) { (title, details) ->
            Text(
                text = title,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 4.dp)
            )

            details.forEach { line ->
                if (line.startsWith("http")) {
                    ClickableText(
                        text = buildAnnotatedString {
                            pushStringAnnotation(tag = "URL", annotation = line)
                            withStyle(style = SpanStyle(color = Color.Blue, textDecoration = TextDecoration.Underline)) {
                                append("• $line")
                            }
                            pop()
                        },
                        modifier = Modifier.padding(vertical = 2.dp),
                        onClick = {
                            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(line))
                            context.startActivity(intent)
                        }
                    )
                } else {
                    Text(
                        text = "• $line",
                        fontSize = 14.sp,
                        modifier = Modifier.padding(vertical = 2.dp)
                    )
                }
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))
        }

        // Additional License Texts with "Read More" toggle
        item {
            var expanded by remember { mutableStateOf(false) }

            Text(
                text = "Apache License 2",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 8.dp)
            )

            if (expanded) {
                Text(
                    text = LicenseStrings.apacheLicense,
                    fontSize = 14.sp
                )
                Divider(modifier = Modifier.padding(vertical = 8.dp))
            }

            TextButton(onClick = { expanded = !expanded }) {
                Text(text = if (expanded) "Read Less" else "Read More")
            }
        }

        item {
            var expanded by remember { mutableStateOf(false) }

            Text(
                text = "MIT License",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 8.dp)
            )

            if (expanded) {
                Text(
                    text = LicenseStrings.mitLicense,
                    fontSize = 14.sp
                )
                Divider(modifier = Modifier.padding(vertical = 8.dp))
            }

            TextButton(onClick = { expanded = !expanded }) {
                Text(text = if (expanded) "Read Less" else "Read More")
            }
        }
    }
}