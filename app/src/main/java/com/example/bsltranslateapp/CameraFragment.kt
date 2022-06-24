package com.example.bsltranslateapp

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.Configuration
import android.hardware.display.DisplayManager
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.*
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import androidx.window.layout.WindowMetricsCalculator
import com.example.bsltranslateapp.databinding.CameraUiContainerBinding
import com.example.bsltranslateapp.databinding.FragmentCameraBinding
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min


typealias VideoOutputListener = (videoText: String) -> Unit

class CameraFragment : Fragment() {
    private val TAG = "CAMERA_FRAGMENT"
    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding get() = _fragmentCameraBinding!!

    private var cameraUiContainerBinding: CameraUiContainerBinding? = null

    private var displayId: Int = -1
    private var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var windowMetricsCalculator: WindowMetricsCalculator

    private val handsOptions = HandsOptions.builder()
        .setStaticImageMode(false)
        .setMaxNumHands(2)
        .setRunOnGpu(false)
        .build()
    private val hands by lazy {
        Hands(requireContext(), handsOptions)
    }

        private val displayManager by lazy {
            requireContext().getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
        }

        private lateinit var cameraExecutor: ExecutorService

        // Display listener for orientation changes
        private val displayListener = object : DisplayManager.DisplayListener {
            override fun onDisplayAdded(displayId: Int) = Unit
            override fun onDisplayRemoved(displayId: Int) = Unit
            override fun onDisplayChanged(displayId: Int) = view?.let { view ->
                if (displayId == this@CameraFragment.displayId) {
                    Log.d(TAG, "Rotation changed: ${view.display.rotation}")
                    imageAnalyzer?.targetRotation = view.display.rotation
                }
            } ?: Unit
        }

        override fun onResume() {
            super.onResume()
            // Make sure permissions still present
            if (!PermissionsFragment.hasPermissions(requireContext())) {
                Navigation.findNavController(requireActivity(), R.id.fragment_container).navigate(
                    CameraFragmentDirections.actionCameraToPermissions()
                )
            }
        }

        override fun onDestroyView() {
            _fragmentCameraBinding = null
            super.onDestroyView()

            cameraExecutor.shutdown()
            displayManager.unregisterDisplayListener(displayListener)
        }

        override fun onCreateView(
            inflater: LayoutInflater,
            container: ViewGroup?,
            savedInstanceState: Bundle?
        ): View {
            _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
            return fragmentCameraBinding.root
        }

        @SuppressLint("MissingPermission")
        override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
            super.onViewCreated(view, savedInstanceState)

            cameraExecutor = Executors.newSingleThreadExecutor()

            // Update rotation when device orientation changes
            displayManager.registerDisplayListener(displayListener, null)

            // Initialize WindowManager for display metrics
            windowMetricsCalculator = WindowMetricsCalculator.getOrCreate()

            // Wait for views to be laid out
            fragmentCameraBinding.viewFinder.post {
                displayId = fragmentCameraBinding.viewFinder.display.displayId
                updateCameraUi()
                setUpCamera()
            }
        }

        override fun onConfigurationChanged(newConfig: Configuration) {
            super.onConfigurationChanged(newConfig)

            // Rebind camera
            bindCameraUseCases()

            // Enable or disable switching between cameras
            updateCameraSwitchButton()
        }

        // Init cameraX and prepare to bind use cases
        private fun setUpCamera() {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
            cameraProviderFuture.addListener({

                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Select lensFacing depending on available cameras
                lensFacing = when {
                    hasFrontCamera() -> CameraSelector.LENS_FACING_FRONT
                    hasBackCamera() -> CameraSelector.LENS_FACING_BACK
                    else -> throw IllegalStateException("Back and front camera are unavailable")
                }

                // Enable or disable switching between cameras
                updateCameraSwitchButton()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext()))
        }

        /** Declare and bind preview, capture and analysis use cases */
        private fun bindCameraUseCases() {

            // Get screen metrics used to setup camera for full screen resolution
            val metrics = windowMetricsCalculator.computeCurrentWindowMetrics(requireActivity()).bounds
            Log.d(TAG, "Screen metrics: ${metrics.width()} x ${metrics.height()}")

            val screenAspectRatio = aspectRatio(metrics.width(), metrics.height())
            Log.d(TAG, "Preview aspect ratio: $screenAspectRatio")

            val rotation = fragmentCameraBinding.viewFinder.display.rotation

            // CameraProvider
            val cameraProvider = cameraProvider
                ?: throw IllegalStateException("Camera initialization failed.")

            // CameraSelector
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Preview
            preview = Preview.Builder()
                // We request aspect ratio but no resolution
                .setTargetAspectRatio(screenAspectRatio)
                // Set initial target rotation
                .setTargetRotation(rotation)
                .build()

            // ImageAnalysis
            imageAnalyzer = ImageAnalysis.Builder()
                // We request aspect ratio but no resolution
                .setTargetAspectRatio(screenAspectRatio)
                // Set initial target rotation, we will have to call this again if rotation changes
                // during the lifecycle of this use case
                .setTargetRotation(rotation)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer(requireContext()) { imageOutput ->
                        writeAnalysisOutput(imageOutput)
                    })
                }

            // Must unbind the use-cases before rebinding them
            cameraProvider.unbindAll()

            try {
                // A variable number of use-cases can be passed here -
                // camera provides access to CameraControl & CameraInfo
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageAnalyzer, preview)

                // Attach the viewfinder's surface provider to preview use case
                preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
                observeCameraState(camera?.cameraInfo!!)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed")
            }
        }

        private fun writeAnalysisOutput(imageOutput: String) {
            val imageOutputBinding = cameraUiContainerBinding?.imageOutput
            imageOutputBinding?.post { imageOutputBinding.text = imageOutput }
        }

        private fun observeCameraState(cameraInfo: CameraInfo) {
            cameraInfo.cameraState.observe(viewLifecycleOwner) { cameraState ->
                run {
                    when (cameraState.type) {
                        CameraState.Type.PENDING_OPEN -> {
                            // Ask the user to close other camera apps
                            Log.w(TAG, "Camera currently in use by another app")
                        }
                        CameraState.Type.OPENING -> {
                            // Show the Camera UI
                            Log.i(TAG, "Camera opening")
                        }
                        CameraState.Type.OPEN -> {
                            // Setup Camera resources and begin processing
                            Log.i(TAG, "Camera open")
                        }
                        CameraState.Type.CLOSING -> {
                            // Close camera UI
                            Log.i(TAG, "Camera closing")
                        }
                        CameraState.Type.CLOSED -> {
                            // Free camera resources
                            Log.i(TAG, "Camera closed")
                        }
                    }
                }

                cameraState.error?.let { error ->
                    when (error.code) {
                        // Open errors
                        CameraState.ERROR_STREAM_CONFIG -> {
                            // Make sure to setup the use cases properly
                            Log.e(TAG, "Stream config error")
                        }
                        // Opening errors
                        CameraState.ERROR_CAMERA_IN_USE -> {
                            // Close the camera or ask user to close another camera app that's using the
                            // camera
                            Log.e(TAG, "Camera in use")
                        }
                        CameraState.ERROR_MAX_CAMERAS_IN_USE -> {
                            // Close another open camera in the app, or ask the user to close another
                            // camera app that's using the camera
                            Log.e(TAG, "Max cameras in use")
                        }
                        CameraState.ERROR_OTHER_RECOVERABLE_ERROR -> {
                            Log.e(TAG, "Other recoverable error")
                        }
                        // Closing errors
                        CameraState.ERROR_CAMERA_DISABLED -> {
                            // Ask the user to enable the device's cameras
                            Log.e(TAG, "Camera disabled")
                        }
                        CameraState.ERROR_CAMERA_FATAL_ERROR -> {
                            // Ask the user to reboot the device to restore camera function
                            Log.e(TAG, "Fatal error")
                        }
                        // Closed errors
                        CameraState.ERROR_DO_NOT_DISTURB_MODE_ENABLED -> {
                            // Ask the user to disable the "Do Not Disturb" mode, then reopen the camera
                            Log.e(TAG, "Do not disturb mode enable")
                        }
                    }
                }
            }
        }

        /**
         *  [androidx.camera.core.ImageAnalysis.Builder] requires enum value of
         *  [androidx.camera.core.AspectRatio]. Currently it has values of 4:3 & 16:9.
         *
         *  Detecting the most suitable ratio for dimensions provided in @params by counting absolute
         *  of preview ratio to one of the provided values.
         *
         *  @param width - preview width
         *  @param height - preview height
         *  @return suitable aspect ratio
         */
        private fun aspectRatio(width: Int, height: Int): Int {
            val previewRatio = max(width, height).toDouble() / min(width, height)
            if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
                return AspectRatio.RATIO_4_3
            }
            return AspectRatio.RATIO_16_9
        }

        /** Method used to re-draw the camera UI controls, called every time configuration changes. */
        private fun updateCameraUi() {

            // Remove previous UI if any
            cameraUiContainerBinding?.root?.let {
                fragmentCameraBinding.root.removeView(it)
            }

            cameraUiContainerBinding = CameraUiContainerBinding.inflate(
                LayoutInflater.from(requireContext()),
                fragmentCameraBinding.root,
                true
            )

            // Setup for button used to switch cameras
            cameraUiContainerBinding?.cameraSwitchButton?.let {

                // Disable the button until the camera is set up
                it.isEnabled = false

                // Listener for button used to switch cameras. Only called if the button is enabled
                it.setOnClickListener {
                    lensFacing = if (CameraSelector.LENS_FACING_FRONT == lensFacing) {
                        CameraSelector.LENS_FACING_BACK
                    } else {
                        CameraSelector.LENS_FACING_FRONT
                    }
                    // Re-bind use cases to update selected camera
                    bindCameraUseCases()
                }
            }
        }

        /** Enabled or disabled a button to switch cameras depending on the available cameras */
        private fun updateCameraSwitchButton() {
            try {
                cameraUiContainerBinding?.cameraSwitchButton?.isEnabled = hasBackCamera() && hasFrontCamera()
            } catch (exception: CameraInfoUnavailableException) {
                cameraUiContainerBinding?.cameraSwitchButton?.isEnabled = false
            }
        }

        /** Returns true if the device has an available back camera. False otherwise */
        private fun hasBackCamera(): Boolean {
            return cameraProvider?.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) ?: false
        }

        /** Returns true if the device has an available front camera. False otherwise */
        private fun hasFrontCamera(): Boolean {
            return cameraProvider?.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) ?: false
        }




        companion object {
        private const val TAG = "CAMERA"
        private const val RATIO_4_3_VALUE = 4.0 / 3.0
        private const val RATIO_16_9_VALUE = 16.0 / 9.0
    }
}