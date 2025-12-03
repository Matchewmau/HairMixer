import React, { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import APIService from "../services/api";
import AuthService from "../services/AuthService";
import Navbar from "../components/Navbar";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import Modal from "../components/ui/Modal";

const PhotoUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [user, setUser] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [showGuideModal, setShowGuideModal] = useState(false);
  const [stream, setStream] = useState(null);

  const navigate = useNavigate();
  const location = useLocation();
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        if (!AuthService.getAccessToken()) {
          setUser(null);
          return;
        }
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error("Authentication check failed:", error);
        setUser(null);
      }
    };

    checkAuth();

    // Check if we should show the guide modal (passed from Analyze)
    if (location.state?.showGuide) {
      setShowGuideModal(true);
      // Clear the state so it doesn't reopen on refresh/navigation
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      setStream(mediaStream);
      setShowCamera(true);

      // Wait for videoRef to be available
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      }, 100);
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert(
        "Could not access camera. Please make sure you have granted camera permissions."
      );
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setShowCamera(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          const file = new File([blob], "camera-photo.jpg", {
            type: "image/jpeg",
          });
          setSelectedFile(file);
          setPreviewUrl(URL.createObjectURL(file));
          stopCamera();
        },
        "image/jpeg",
        0.95
      );
    }
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  const handleFiles = (files) => {
    const file = files[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      alert("Please select a valid image file");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const performAnalysis = async () => {
    if (!selectedFile) {
      alert("Please select an image first");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      // Upload image to backend
      const response = await APIService.uploadImage(selectedFile);

      console.log("Upload response:", response);

      // Check if face was detected
      if (response.face_detected) {
        setAnalysisResult("success");

        // Navigate directly to preferences form after 1.5 seconds
        setTimeout(() => {
          navigate("/preferences", {
            state: {
              imageFile: selectedFile,
              previewUrl: previewUrl,
              uploadResponse: response,
            },
          });
        }, 1500);
      } else {
        setAnalysisResult("failed");
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      setAnalysisResult("failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile) {
      // If no file selected, show guide modal instead of alert
      setShowGuideModal(true);
      return;
    }
    performAnalysis();
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar transparent={true} user={user} onLogout={handleLogout} />

      <div className="min-h-screen bg-background pt-20 md:pt-24 relative overflow-hidden">
        {/* Background Elements */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/20 rounded-full blur-3xl opacity-30 animate-pulse-slow pointer-events-none"></div>
        <div
          className="absolute bottom-0 left-0 w-96 h-96 bg-secondary/20 rounded-full blur-3xl opacity-30 animate-pulse-slow pointer-events-none"
          style={{ animationDelay: "1s" }}
        ></div>

        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
          {/* Header */}
          <div className="text-center mb-8 md:mb-12 animate-fade-in">
            <div className="mb-6 animate-slide-up">
              <span className="inline-block bg-primary/10 text-primary-foreground px-4 py-1.5 rounded-full text-sm font-medium border border-primary/20 backdrop-blur-md">
                Step 1: Photo Analysis
              </span>
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-heading font-bold mb-4 md:mb-6 text-white">
              Upload Your Photo
            </h1>
            <p className="text-lg md:text-xl text-gray-400 max-w-3xl mx-auto leading-relaxed mb-6">
              Upload a clear photo of your face or use your camera to get
              personalized hairstyle recommendations
            </p>
            <Button
              onClick={() => setShowGuideModal(true)}
              variant="outline"
              className="gap-2"
            >
              <span>ℹ️</span> View Photo Guidelines
            </Button>
          </div>

          {/* Camera Modal */}
          {showCamera && (
            <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
              <div className="bg-surface border border-white/10 rounded-2xl max-w-4xl w-full shadow-2xl overflow-hidden">
                <div className="flex items-center justify-between p-4 md:p-6 border-b border-white/10">
                  <h2 className="text-xl md:text-2xl font-heading font-bold text-white">
                    Take a Photo
                  </h2>
                  <button
                    onClick={stopCamera}
                    className="text-gray-400 hover:text-white transition-colors duration-300 p-2 hover:bg-white/10 rounded-lg"
                  >
                    <svg
                      className="w-6 h-6"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>

                <div className="p-4 md:p-6">
                  <div className="relative bg-black rounded-xl overflow-hidden mb-4 md:mb-6 border border-white/10">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-auto max-h-[60vh] object-contain"
                    />
                  </div>

                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button
                      onClick={capturePhoto}
                      variant="primary"
                      size="lg"
                      className="shadow-lg shadow-primary/20"
                    >
                      📸 Capture Photo
                    </Button>
                    <Button onClick={stopCamera} variant="ghost" size="lg">
                      Cancel
                    </Button>
                  </div>
                </div>
              </div>
              <canvas ref={canvasRef} className="hidden" />
            </div>
          )}

          {/* Upload Area */}
          <Card
            className="p-6 md:p-8 mb-8 animate-slide-up"
            style={{ animationDelay: "0.1s" }}
          >
            <div
              className={`border-2 border-dashed rounded-xl p-8 md:p-12 text-center transition-all duration-300 ${
                dragActive
                  ? "border-primary bg-primary/10 transform scale-105"
                  : selectedFile
                  ? "border-green-500/50 bg-green-500/5"
                  : "border-white/10 hover:border-primary/50 hover:bg-white/5"
              }`}
              onDragEnter={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setDragActive(false);
              }}
              onDragOver={(e) => {
                e.preventDefault();
              }}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <div className="space-y-8">
                  <div className="relative inline-block group">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="mx-auto h-48 w-48 md:h-64 md:w-64 object-cover rounded-2xl shadow-2xl border-2 border-primary/30 transition-transform duration-500 group-hover:scale-105"
                    />
                    <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 bg-green-500/90 backdrop-blur-md text-white px-4 py-1 rounded-full text-sm font-bold shadow-lg flex items-center gap-2 whitespace-nowrap">
                      <span>✓</span> Photo Selected
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      variant="outline"
                    >
                      Choose different photo
                    </Button>
                    <Button onClick={startCamera} variant="secondary">
                      📸 Use Camera Instead
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-8">
                  <div className="relative inline-block">
                    <div className="w-20 h-20 md:w-24 md:h-24 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-primary/20">
                      <svg
                        className="h-10 w-10 md:h-12 md:w-12 text-primary"
                        stroke="currentColor"
                        fill="none"
                        viewBox="0 0 48 48"
                      >
                        <path
                          d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                          strokeWidth={2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </div>
                  </div>
                  <div className="text-gray-300">
                    <p className="text-xl md:text-2xl font-heading font-bold text-white mb-2">
                      Drop your photo here
                    </p>
                    <p className="text-base md:text-lg text-gray-400">
                      or choose an option below
                    </p>
                  </div>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center items-center max-w-md mx-auto">
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      variant="primary"
                      className="w-full sm:w-auto flex-1 shadow-lg shadow-primary/20"
                      size="lg"
                    >
                      📁 Browse Files
                    </Button>
                    <Button
                      onClick={startCamera}
                      variant="secondary"
                      className="w-full sm:w-auto flex-1"
                      size="lg"
                    >
                      📸 Use Camera
                    </Button>
                  </div>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => handleFiles(e.target.files)}
              className="hidden"
            />

            {/* Photo Tips */}
            <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 md:gap-6">
              {[
                { icon: "✓", text: "Good lighting" },
                { icon: "✓", text: "Clear face view" },
                { icon: "✓", text: "No sunglasses" },
              ].map((tip, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-center space-x-3 text-gray-300 bg-surface/50 rounded-xl p-4 border border-white/5"
                >
                  <span className="text-green-400 font-bold text-lg bg-green-400/10 w-8 h-8 rounded-full flex items-center justify-center">
                    {tip.icon}
                  </span>
                  <span className="font-medium">{tip.text}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Analyze Button */}
          <div
            className="text-center mb-8 md:mb-12 animate-slide-up"
            style={{ animationDelay: "0.2s" }}
          >
            <Button
              onClick={handleAnalyzeClick}
              disabled={!selectedFile || isAnalyzing}
              variant="primary"
              size="lg"
              className="px-12 py-4 text-xl shadow-xl shadow-primary/25"
              isLoading={isAnalyzing}
              loadingText="Analyzing Face..."
            >
              Analyze My Face
            </Button>
          </div>

          {/* Analysis Status */}
          {analysisResult === "success" && (
            <Card className="border-green-500/30 bg-green-500/5 text-center animate-fade-in p-8">
              <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                <div className="text-green-400 text-4xl font-bold">✓</div>
              </div>
              <p className="text-green-300 font-heading font-bold text-2xl mb-2">
                Face detected successfully!
              </p>
              <p className="text-gray-400">Redirecting to preferences...</p>
            </Card>
          )}

          {analysisResult === "failed" && (
            <Card className="border-red-500/30 bg-red-500/5 text-center animate-fade-in p-8">
              <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                <div className="text-red-400 text-4xl font-bold">✗</div>
              </div>
              <p className="text-red-300 font-heading font-bold text-2xl mb-4">
                Could not detect a face
              </p>
              <p className="text-red-400/80 mb-8 max-w-md mx-auto">
                Please try again with a clearer photo where your face is clearly
                visible and well-lit.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setAnalysisResult(null);
                  }}
                  variant="primary"
                >
                  Try Another Photo
                </Button>
                <Button onClick={startCamera} variant="secondary">
                  📸 Use Camera
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
      {/* Photo Guide Modal */}
      <Modal
        isOpen={showGuideModal}
        onClose={() => setShowGuideModal(false)}
        title="Photo Tips for Best Results"
        size="lg"
      >
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-black/20 rounded-xl p-4 border border-white/10 h-full flex flex-col">
                <p className="text-sm text-gray-400 mb-3 text-center font-medium">
                  Ideal Photo Example
                </p>
                <div className="flex-1 flex items-center justify-center bg-black/40 rounded-lg overflow-hidden">
                  <img
                    src="/upload/pic_guide.png"
                    alt="Good photo guide"
                    className="w-full h-auto object-contain max-h-[300px]"
                  />
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="bg-black/20 rounded-xl p-4 border border-white/10 h-full flex flex-col">
                <p className="text-sm text-gray-400 mb-3 text-center font-medium">
                  What to Avoid (Accessories)
                </p>
                <div className="flex-1 flex items-center justify-center bg-black/40 rounded-lg overflow-hidden">
                  <img
                    src="/upload/remove_guide.png"
                    alt="Remove accessories guide"
                    className="w-full h-auto object-contain max-h-[300px]"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-surface/50 rounded-xl p-5 border border-white/5">
            <h3 className="text-lg font-bold text-white mb-4">
              Checklist for Success
            </h3>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                "Look straight at the camera",
                "Avoid a side profile of your face",
                "Find the best spot with good lighting",
                "Keep your face fully visible",
                "Remove glasses, mask, and hat",
                "Make sure hair does not cover the face",
              ].map((tip, idx) => (
                <li key={idx} className="flex items-start gap-3 text-gray-300">
                  <span className="text-primary mt-0.5 font-bold">✓</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="flex justify-end pt-2 border-t border-white/10">
            <Button onClick={() => setShowGuideModal(false)} variant="primary">
              Got it, I'm ready!
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default PhotoUpload;
