package ai.mlc.mlcchat

import android.app.Activity
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import org.json.JSONObject

@ExperimentalMaterial3Api
@Composable
fun NavView(activity: Activity, appViewModel: AppViewModel = viewModel()) {
    val navController = rememberNavController()
    val json = (activity as MainActivity).assets.open("inputs.json").reader().readText()
    val data = JSONObject(json).getJSONArray("data")
    NavHost(navController = navController, startDestination = "home") {
        composable("home") { StartView(navController, appViewModel) }
        composable("chat") { ChatView(navController, appViewModel.chatState, activity, data) }
    }
}