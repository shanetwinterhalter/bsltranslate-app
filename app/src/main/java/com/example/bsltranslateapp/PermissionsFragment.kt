package com.example.bsltranslateapp

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation

private val PERMISSIONS_REQUIRED = arrayOf(Manifest.permission.CAMERA)
private const val TAG = "PERMISSIONS"

class PermissionsFragment : Fragment() {

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        permissions.forEach { actionMap ->
            when (actionMap.key) {
                Manifest.permission.CAMERA -> {
                    if (actionMap.value) {
                        navigateToCamera()
                        Log.i(TAG, "Permission granted")
                    } else {
                        Log.e(TAG, "Permission request denied")
                        Toast.makeText(context, "This app requires use of the camera", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!hasPermissions(requireContext())) {
            requestPermission.launch(PERMISSIONS_REQUIRED)
        } else {
            Log.i(TAG, "Required permissions already present")
            navigateToCamera()
        }
    }

    private fun navigateToCamera() {
        lifecycleScope.launchWhenStarted {
            Navigation.findNavController(requireActivity(), R.id.fragment_container).navigate(
                PermissionsFragmentDirections.actionPermissionsToCamera())
        }
    }

    companion object {
        fun hasPermissions(context: Context) = PERMISSIONS_REQUIRED.all {
            ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}