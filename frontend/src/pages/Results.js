import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import AuthService from "../services/AuthService";
import apiService from "../services/api";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import Modal from "../components/ui/Modal";

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [uploadResponse, setUploadResponse] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [preferences, setPreferences] = useState(null);

  // Modal states
  const [showTryHairstyleModal, setShowTryHairstyleModal] = useState(false);
  const [currentHairstyleIndex, setCurrentHairstyleIndex] = useState(0);
  const [hairstyleDetails, setHairstyleDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [detailsError, setDetailsError] = useState("");
  const [showImageModal, setShowImageModal] = useState(false);
  const [activeImage, setActiveImage] = useState(null);

  // Cache for hairstyle details to avoid refetching
  const [hairstyleDetailsCache, setHairstyleDetailsCache] = useState({});

  // Abort controller for overlay generation
  const [overlayAbortController, setOverlayAbortController] = useState(null);

  // Saved hairstyles tracking
  const [savedThisSession, setSavedThisSession] = useState(new Set());

  useEffect(() => {
    // Check if we have state from previous pages
    if (!location.state) {
      navigate("/upload");
      return;
    }

    const {
      recommendations: recs,
      uploadResponse: uploadResp,
      previewUrl: url,
      imageFile: file,
      preferences: prefs,
    } = location.state;

    if (!recs || !uploadResp) {
      navigate("/upload");
      return;
    }

    setRecommendations(recs);
    setUploadResponse(uploadResp);
    setPreviewUrl(url);
    setImageFile(file);
    setPreferences(prefs);

    checkAuth();
  }, [location, navigate]);

  const checkAuth = async () => {
    try {
      const currentUser = await AuthService.getCurrentUser();
      setUser(currentUser);
    } catch (error) {
      console.error("Authentication check failed:", error);
      setUser(null);
    }
  };

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const resolveMediaUrl = (url) => {
    if (!url) return "";
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    const serverOrigin = apiService.baseURL.replace(/\/api\/?$/, "");
    return `${serverOrigin}${url}`;
  };

  const handleTryHairstyle = async (index) => {
    try {
      setDetailsError("");
      setCurrentHairstyleIndex(index);
      setShowTryHairstyleModal(true);

      const style = recommendations.recommendations[index];
      const cacheKey = style.id;

      // Check if we already have cached details for this hairstyle
      if (hairstyleDetailsCache[cacheKey]) {
        console.log("Using cached hairstyle details for:", style.name);
        setHairstyleDetails(hairstyleDetailsCache[cacheKey]);
        return;
      }

      // If not cached, fetch from API
      setLoadingDetails(true);

      // Fetch detailed information
      const details = await apiService.getHairstyleDetailsWithAI(
        style.id,
        preferences?.id || null,
        uploadResponse?.image_id || null
      );

      // Generate overlay with AbortController for cancellation
      if (uploadResponse?.image_id && style?.id) {
        try {
          // Create new AbortController for this overlay request
          const controller = new AbortController();
          setOverlayAbortController(controller);

          const resp = await apiService.generateOverlay(
            uploadResponse.image_id,
            style.id,
            "advanced",
            controller.signal, // Pass the abort signal
            true // Use hair color from user preferences
          );
          details.overlay_url = resolveMediaUrl(resp.overlay_url);

          // Clear the abort controller after successful completion
          setOverlayAbortController(null);
        } catch (overlayError) {
          if (overlayError.name === "AbortError") {
            console.log("Overlay generation was cancelled");
            setDetailsError("Overlay generation was cancelled");
            setLoadingDetails(false);
            setOverlayAbortController(null);
            return;
          }
          console.warn("Overlay generation failed:", overlayError);
          details.overlay_url = null;
          setOverlayAbortController(null);
        }
      }

      setHairstyleDetails(details);

      // Cache the details for future use
      setHairstyleDetailsCache((prev) => ({
        ...prev,
        [cacheKey]: details,
      }));
    } catch (e) {
      console.error("Failed to load hairstyle details:", e);
      setDetailsError(e?.message || "Failed to load hairstyle details");
    } finally {
      setLoadingDetails(false);
      setOverlayAbortController(null);
    }
  };

  const cancelOverlayGeneration = () => {
    if (overlayAbortController) {
      console.log("Cancelling overlay generation...");
      overlayAbortController.abort();
      setOverlayAbortController(null);
      setLoadingDetails(false);
      setShowTryHairstyleModal(false);
      setHairstyleDetails(null);
      setDetailsError("");
    }
  };

  const handleNextHairstyle = () => {
    const nextIndex =
      (currentHairstyleIndex + 1) % recommendations.recommendations.length;
    handleTryHairstyle(nextIndex);
  };

  const handlePrevHairstyle = () => {
    const prevIndex =
      currentHairstyleIndex === 0
        ? recommendations.recommendations.length - 1
        : currentHairstyleIndex - 1;
    handleTryHairstyle(prevIndex);
  };

  const closeTryHairstyleModal = () => {
    setShowTryHairstyleModal(false);
    setHairstyleDetails(null);
    setDetailsError("");
  };

  const openImageModal = (imageUrl) => {
    setActiveImage(imageUrl);
    setShowImageModal(true);
  };

  const saveHairstyleRecommendation = async (
    hairstyleId,
    hairstyleName,
    hairstyleData,
    details = null
  ) => {
    if (!user) {
      // Prompt for login if not authenticated
      alert("Please log in to save hairstyles.");
      return;
    }

    try {
      // Optimistically update UI
      setSavedThisSession((prev) => new Set(prev).add(hairstyleId));

      // Prepare payload
      const payload = {
        hairstyle_id: hairstyleId,
        hairstyle_name: hairstyleName,
        recommendation_data: {
          ...hairstyleData,
          styling_tips: details?.styling_tips || [],
          maintenance_guide: details?.maintenance_guide || [],
          products: details?.products || [],
          preference_match: details?.preference_match || [],
          original_image: previewUrl || null,
        },
        user_preferences: preferences || {},
        face_shape: uploadResponse?.face_shape?.shape,
        face_shape_confidence: uploadResponse?.face_shape?.confidence,
        overlay_url: details?.overlay_url || null,
        personalized_description: details?.personalized_description || null,
      };

      console.log("Saving hairstyle payload:", payload);

      await apiService.saveHairstyle(payload);
    } catch (error) {
      console.error("Failed to save hairstyle:", error);
      alert("Failed to save hairstyle. Please try again.");
      // Revert optimistic update
      setSavedThisSession((prev) => {
        const newSet = new Set(prev);
        newSet.delete(hairstyleId);
        return newSet;
      });
    }
  };

  if (!recommendations) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center p-8">
          <h2 className="text-3xl font-heading font-bold text-white mb-6">
            No recommendations found
          </h2>
          <Button
            onClick={() => navigate("/upload")}
            variant="primary"
            size="lg"
          >
            Start Over
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background pt-20 md:pt-24 pb-12">
      <Navbar user={user} onLogout={handleLogout} />

      {/* Header Section */}
      <div className="relative mb-12">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 blur-3xl opacity-30 pointer-events-none"></div>
        <div className="max-w-7xl mx-auto px-4 text-center relative z-10">
          <h1 className="text-4xl md:text-5xl font-heading font-bold text-white mb-4 animate-fade-in">
            Your Hairstyle Recommendations
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto animate-slide-up">
            Based on your preferences and facial analysis
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 space-y-12">
        {/* Face Analysis Summary */}
        {uploadResponse && (
          <Card
            className="p-8 animate-slide-up"
            style={{ animationDelay: "0.1s" }}
          >
            <h2 className="text-2xl font-heading font-bold text-white mb-6 border-b border-white/10 pb-4">
              Face Analysis
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              {/* User Image */}
              {previewUrl && (
                <div className="flex justify-center">
                  <div className="relative w-full max-w-xs mx-auto group">
                    <div className="aspect-square overflow-hidden rounded-2xl shadow-2xl border-2 border-primary/30 relative">
                      <img
                        src={previewUrl}
                        alt="Uploaded face for analysis"
                        className="w-full h-full object-cover object-center transition-transform duration-500 group-hover:scale-105"
                        style={{ objectPosition: "center 30%" }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-4">
                        <span className="text-white font-medium">
                          Original Photo
                        </span>
                      </div>
                    </div>
                    <div className="absolute top-4 right-4 bg-primary/90 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">
                      Your Photo
                    </div>
                  </div>
                </div>
              )}

              {/* Face Shape Info */}
              <div className="space-y-6">
                <div className="bg-primary/10 border border-primary/20 rounded-xl p-6 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-primary/20 rounded-full blur-2xl -mr-10 -mt-10"></div>
                  <h3 className="font-medium text-gray-300 mb-2 text-lg relative z-10">
                    Detected Face Shape
                  </h3>
                  <p className="text-primary-foreground font-heading font-bold text-4xl capitalize relative z-10">
                    {uploadResponse.face_shape?.shape || "Unknown"}
                  </p>
                  {uploadResponse.face_shape?.confidence && (
                    <div className="mt-2 text-sm text-primary/80 font-medium relative z-10 hidden">
                      {Math.round(uploadResponse.face_shape.confidence * 100)}%
                      Confidence
                    </div>
                  )}
                </div>

                {/* Face shape characteristics */}
                {uploadResponse.face_shape?.shape && (
                  <div className="p-6 bg-surface/50 border border-white/5 rounded-xl">
                    <h4 className="font-heading font-bold text-white mb-4 text-lg">
                      Characteristics & Tips
                    </h4>
                    <div className="text-gray-300 space-y-3 leading-relaxed">
                      {/* Content logic remains same, just styling updated */}
                      {uploadResponse.face_shape.shape.toLowerCase() ===
                        "oval" && (
                        <>
                          <p>
                            <strong className="text-white">Proportions:</strong>{" "}
                            Well-balanced with slightly wider cheekbones than
                            forehead and jawline.
                          </p>
                          <p>
                            <strong className="text-white">
                              Best suited for:
                            </strong>{" "}
                            Almost all hairstyles work beautifully. You have the
                            most versatile canvas!
                          </p>
                        </>
                      )}
                      {uploadResponse.face_shape.shape.toLowerCase() ===
                        "round" && (
                        <>
                          <p>
                            <strong className="text-white">Proportions:</strong>{" "}
                            Soft curves with similar width and length, fuller
                            cheeks.
                          </p>
                          <p>
                            <strong className="text-white">
                              Best suited for:
                            </strong>{" "}
                            Styles with height/volume at the crown to elongate.
                            Angular cuts work well.
                          </p>
                        </>
                      )}
                      {uploadResponse.face_shape.shape.toLowerCase() ===
                        "square" && (
                        <>
                          <p>
                            <strong className="text-white">Proportions:</strong>{" "}
                            Strong, defined jawline with equal width
                            forehead/jaw.
                          </p>
                          <p>
                            <strong className="text-white">
                              Best suited for:
                            </strong>{" "}
                            Soft, layered styles and waves to soften angular
                            features.
                          </p>
                        </>
                      )}
                      {uploadResponse.face_shape.shape.toLowerCase() ===
                        "heart" && (
                        <>
                          <p>
                            <strong className="text-white">Proportions:</strong>{" "}
                            Wider forehead tapering to a narrow chin.
                          </p>
                          <p>
                            <strong className="text-white">
                              Best suited for:
                            </strong>{" "}
                            Chin-length bobs, side-swept bangs, volume at
                            jawline.
                          </p>
                        </>
                      )}
                      {/* Fallback for other shapes */}
                      {!["oval", "round", "square", "heart"].includes(
                        uploadResponse.face_shape.shape.toLowerCase()
                      ) && (
                        <p>
                          <strong className="text-white">
                            Best suited for:
                          </strong>{" "}
                          Most hairstyles work well with your unique face shape!
                        </p>
                      )}
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap gap-4 pt-2">
                  <Button
                    variant="outline"
                    onClick={() => navigate("/upload")}
                    className="flex-1"
                  >
                    Try Another Photo
                  </Button>
                  <Button
                    variant="primary"
                    onClick={() =>
                      navigate("/preferences", {
                        state: {
                          imageFile,
                          previewUrl,
                          uploadResponse,
                          existingPreferences: preferences,
                        },
                      })
                    }
                    className="flex-1"
                  >
                    Update Preferences
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Recommendations Grid */}
        <div className="animate-slide-up" style={{ animationDelay: "0.2s" }}>
          <h2 className="text-3xl font-heading font-bold text-white mb-8 flex items-center gap-3">
            <span className="bg-gradient-to-r from-primary to-secondary w-2 h-8 rounded-full"></span>
            Recommended Hairstyles
          </h2>

          {recommendations.recommendations &&
          recommendations.recommendations.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {recommendations.recommendations.map((style, index) => (
                <Card
                  key={index}
                  hover
                  className="group flex flex-col h-full overflow-hidden border-white/5 bg-surface/40"
                >
                  <div className="p-6 flex-1 flex flex-col">
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="font-heading font-bold text-2xl text-white group-hover:text-primary transition-colors">
                        {style.name}
                      </h3>
                      {style.match_score > 0 &&
                        Math.round(style.match_score * 100) >= 50 && (
                          <span className="bg-green-500/20 text-green-400 border border-green-500/30 text-xs font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                            Top Match
                          </span>
                        )}
                    </div>

                    <p className="text-gray-400 mb-6 line-clamp-3 leading-relaxed flex-1">
                      {style.description}
                    </p>

                    <Button
                      onClick={() => handleTryHairstyle(index)}
                      variant="primary"
                      className="w-full shadow-lg shadow-primary/20"
                    >
                      Try Hairstyle & View Details
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-20 bg-surface/30 rounded-2xl border border-white/5">
              <div className="text-6xl mb-6">🔍</div>
              <h3 className="text-2xl font-heading font-bold text-white mb-4">
                No Recommendations Found
              </h3>
              <p className="text-gray-400 mb-8 max-w-md mx-auto">
                We couldn't find perfect matches based on your current criteria.
                Try adjusting your preferences.
              </p>
              <Button
                onClick={() => navigate("/preferences")}
                variant="secondary"
              >
                Adjust Preferences
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Try Hairstyle Modal */}
      <Modal
        isOpen={showTryHairstyleModal}
        onClose={closeTryHairstyleModal}
        title={
          recommendations?.recommendations[currentHairstyleIndex]?.name ||
          "Hairstyle Details"
        }
        size="full"
        bodyClassName="p-0 h-full flex flex-col overflow-hidden"
      >
        {loadingDetails ? (
          <div className="flex flex-col items-center justify-center py-20 h-full">
            <div className="relative w-20 h-20 mb-8">
              <div className="absolute inset-0 border-4 border-white/10 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-primary rounded-full border-t-transparent animate-spin"></div>
            </div>
            <p className="text-xl font-heading font-medium text-white mb-2">
              Generating Your New Look...
            </p>
            <p className="text-gray-400 mb-8">
              Using AI to apply the hairstyle to your photo
            </p>
            <Button
              onClick={cancelOverlayGeneration}
              variant="danger"
              variantType="outline"
            >
              Cancel
            </Button>
          </div>
        ) : detailsError ? (
          <div className="text-center py-20 h-full flex flex-col items-center justify-center">
            <div className="text-red-500 text-5xl mb-6">⚠️</div>
            <h3 className="text-xl font-bold text-white mb-2">
              Generation Failed
            </h3>
            <p className="text-red-400 mb-8">{detailsError}</p>
            <Button onClick={closeTryHairstyleModal} variant="secondary">
              Close
            </Button>
          </div>
        ) : hairstyleDetails &&
          recommendations?.recommendations[currentHairstyleIndex] ? (
          <div className="flex flex-col h-full md:h-[80vh]">
            {/* Toolbar - Top Actions */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center px-4 sm:px-6 pt-4 sm:pt-6 pb-4 border-b border-white/10 flex-shrink-0 gap-3 sm:gap-0">
              <div className="flex items-center gap-4">
                <span className="text-gray-400 text-sm font-medium">
                  Style {currentHairstyleIndex + 1} of{" "}
                  {recommendations.recommendations.length}
                </span>
              </div>
              <div className="flex gap-3 w-full sm:w-auto">
                <Button
                  onClick={() =>
                    saveHairstyleRecommendation(
                      recommendations.recommendations[currentHairstyleIndex].id,
                      recommendations.recommendations[currentHairstyleIndex]
                        .name,
                      recommendations.recommendations[currentHairstyleIndex],
                      hairstyleDetails
                    )
                  }
                  variant={
                    savedThisSession.has(
                      recommendations.recommendations[currentHairstyleIndex].id
                    )
                      ? "success"
                      : "outline"
                  }
                  size="sm"
                  className="flex-1 sm:flex-none min-w-[120px] gap-2 justify-center"
                >
                  {savedThisSession.has(
                    recommendations.recommendations[currentHairstyleIndex].id
                  ) ? (
                    <>
                      <span>✓</span> Saved
                    </>
                  ) : (
                    <>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
                        <polyline points="17 21 17 13 7 13 7 21"></polyline>
                        <polyline points="7 3 7 8 15 8"></polyline>
                      </svg>
                      Save Style
                    </>
                  )}
                </Button>
              </div>
            </div>

            <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-6 overflow-y-auto lg:overflow-hidden px-4 sm:px-6 py-4">
              {/* Left Column: Visuals */}
              <div className="lg:w-5/12 flex flex-col gap-4 h-auto lg:h-full flex-shrink-0">
                {/* Before & After Comparison */}
                {hairstyleDetails.overlay_url && previewUrl && (
                  <div className="bg-surface/30 rounded-xl p-4 border border-white/5 shadow-inner flex flex-col h-[300px] sm:h-[400px] lg:h-full">
                    <h3 className="text-sm font-heading font-bold text-white mb-3 flex items-center gap-2 border-b border-white/5 pb-2 flex-shrink-0">
                      <span className="text-lg">✨</span> Transformation
                    </h3>
                    <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
                      <div className="relative rounded-lg overflow-hidden border border-white/10 bg-black/50 group cursor-pointer h-full">
                        <img
                          src={previewUrl}
                          alt="Before"
                          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                          onClick={() => openImageModal(previewUrl)}
                        />
                        <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-white border border-white/10 tracking-wider">
                          BEFORE
                        </div>
                      </div>
                      <div className="relative rounded-lg overflow-hidden border-2 border-primary/50 bg-black/50 shadow-lg shadow-primary/10 group cursor-pointer h-full">
                        <img
                          src={hairstyleDetails.overlay_url}
                          alt="After"
                          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                          onClick={() =>
                            openImageModal(hairstyleDetails.overlay_url)
                          }
                        />
                        <div className="absolute top-2 left-2 bg-primary/90 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-white shadow-lg tracking-wider">
                          AFTER
                        </div>
                      </div>
                    </div>
                    <p className="text-center text-[10px] text-gray-500 mt-2 font-medium uppercase tracking-wide opacity-70 flex-shrink-0">
                      Click to enlarge
                    </p>
                  </div>
                )}

                {/* Reference Image (Fallback) */}
                {!hairstyleDetails.overlay_url &&
                  hairstyleDetails.hairstyle?.image_url && (
                    <div className="bg-surface/30 rounded-xl p-4 border border-white/5 h-[300px] lg:h-full">
                      <h3 className="text-sm font-heading font-bold text-white mb-2">
                        Style Reference
                      </h3>
                      <div className="h-full rounded-lg overflow-hidden shadow-lg">
                        <img
                          src={hairstyleDetails.hairstyle.image_url}
                          alt="Reference"
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </div>
                  )}
              </div>

              {/* Right Column: Details (Scrollable) */}
              <div className="lg:w-7/12 overflow-y-visible lg:overflow-y-auto pr-0 lg:pr-2 space-y-4 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent pb-6 lg:pb-2">
                {/* Description */}
                <Card className="bg-surface/30 border-white/5 p-5">
                  <h3 className="text-md font-heading font-bold text-primary mb-2">
                    Why This Style Works
                  </h3>
                  <p className="text-gray-300 text-sm leading-relaxed text-justify">
                    {hairstyleDetails.personalized_description ||
                      "No description available."}
                  </p>
                </Card>

                {/* Face Shape Match */}
                {(uploadResponse?.face_shape?.shape ||
                  hairstyleDetails.face_shape) && (
                  <Card className="bg-surface/30 border-white/5 p-5">
                    <h3 className="text-md font-heading font-bold text-white mb-2 flex items-center gap-2">
                      <span>👤</span> Face Shape Match
                    </h3>
                    <p className="text-gray-300 text-sm leading-relaxed text-justify">
                      <strong className="text-white block mb-1">
                        Perfect for your{" "}
                        <span className="capitalize text-primary">
                          {uploadResponse?.face_shape?.shape ||
                            hairstyleDetails.face_shape}
                        </span>{" "}
                        face
                      </strong>
                      This hairstyle helps balance your facial proportions,
                      highlights your best features, and creates a harmonious
                      overall look.
                    </p>
                  </Card>
                )}

                {/* Preference Match */}
                {hairstyleDetails.preference_match &&
                  hairstyleDetails.preference_match.length > 0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>✨</span> Why it fits you
                      </h3>
                      <ul className="space-y-3">
                        {hairstyleDetails.preference_match.map((match, idx) => (
                          <li
                            key={idx}
                            className="flex gap-3 text-gray-300 text-sm"
                          >
                            <span className="text-primary font-bold text-lg leading-none mt-0.5">
                              •
                            </span>
                            <span className="text-justify leading-relaxed">
                              {match}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </Card>
                  )}

                {/* Styling Tips */}
                {hairstyleDetails.styling_tips &&
                  hairstyleDetails.styling_tips.length > 0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>💡</span> Pro Styling Tips
                      </h3>
                      <ul className="space-y-3">
                        {hairstyleDetails.styling_tips.map((tip, idx) => (
                          <li
                            key={idx}
                            className="flex gap-3 text-gray-300 text-sm"
                          >
                            <span className="text-primary font-bold text-lg leading-none mt-0.5">
                              •
                            </span>
                            <span className="text-justify leading-relaxed">
                              {tip}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </Card>
                  )}

                {/* Maintenance */}
                {hairstyleDetails.maintenance_guide &&
                  hairstyleDetails.maintenance_guide.length > 0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>🔧</span> Maintenance Guide
                      </h3>
                      <div className="space-y-4">
                        {hairstyleDetails.maintenance_guide.map((step, idx) => (
                          <div key={idx} className="flex gap-3">
                            <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0 mt-0.5 border border-white/10">
                              {idx + 1}
                            </div>
                            <p className="text-gray-300 text-sm leading-relaxed text-justify">
                              {step}
                            </p>
                          </div>
                        ))}
                      </div>
                    </Card>
                  )}

                {/* Recommended Products */}
                {hairstyleDetails.products &&
                  hairstyleDetails.products.length > 0 && (
                    <Card className="bg-surface/30 border-white/5 p-5">
                      <h3 className="text-md font-heading font-bold text-white mb-3 flex items-center gap-2">
                        <span>🧴</span> Recommended Products
                      </h3>
                      <ul className="space-y-3">
                        {hairstyleDetails.products.map((product, idx) => (
                          <li
                            key={idx}
                            className="flex gap-3 text-gray-300 text-sm"
                          >
                            <span className="text-primary font-bold text-lg leading-none mt-0.5">
                              •
                            </span>
                            <span className="text-justify leading-relaxed">
                              {product}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </Card>
                  )}
              </div>
            </div>

            {/* Footer Navigation - Static at bottom of flex container */}
            <div className="flex justify-between items-center border-t border-white/10 px-6 py-4 flex-shrink-0">
              <Button
                onClick={handlePrevHairstyle}
                variant="ghost"
                className="gap-2 text-gray-400 hover:text-white"
              >
                ← Previous
              </Button>

              <Button
                onClick={handleNextHairstyle}
                variant="primary"
                className="gap-2 shadow-lg shadow-primary/20"
              >
                Next →
              </Button>
            </div>
          </div>
        ) : null}
      </Modal>

      {/* Full Screen Image Modal */}
      <Modal
        isOpen={showImageModal}
        onClose={() => setShowImageModal(false)}
        size="lg"
        title="Detailed View"
      >
        {activeImage && (
          <div className="flex items-center justify-center p-4">
            <img
              src={activeImage}
              alt="Full screen view"
              className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl"
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Results;
