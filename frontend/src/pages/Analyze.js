import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import AuthService from "../services/AuthService";
import DiscoverSection from "../components/DiscoverSection";
import Navbar from "../components/Navbar";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";

const Analyze = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
        } else {
          navigate("/login");
        }
      } catch (error) {
        console.error("Authentication check failed:", error);
        navigate("/login");
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <Navbar transparent={true} user={user} onLogout={handleLogout} />

      {/* Full Screen Hero Section */}
      <div className="min-h-screen bg-background pt-20 md:pt-24">
        <div className="bg-gradient-to-br from-background via-surface to-blue-900/20 min-h-screen overflow-hidden">
          <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
            {/* Dark geometric pattern background */}
            <div className="absolute inset-0 opacity-10 mt-20 pointer-events-none">
              <div className="absolute top-0 right-0 w-96 h-96">
                <div className="w-full h-full rounded-full border-2 border-blue-400 transform translate-x-48 -translate-y-48"></div>
              </div>
              <div className="absolute top-1/4 left-0 w-64 h-64">
                <div className="w-full h-full rounded-full border-2 border-purple-400 transform -translate-x-32"></div>
              </div>
              <div className="absolute bottom-0 right-1/3 w-80 h-80">
                <div className="w-full h-full rounded-full border-2 border-indigo-400 transform translate-y-40"></div>
              </div>
              {/* Mesh pattern overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent via-blue-500/5 to-purple-500/5"></div>
              <svg
                className="absolute inset-0 w-full h-full"
                xmlns="http://www.w3.org/2000/svg"
              >
                <defs>
                  <pattern
                    id="grid"
                    width="60"
                    height="60"
                    patternUnits="userSpaceOnUse"
                  >
                    <path
                      d="M 60 0 L 0 0 0 60"
                      fill="none"
                      stroke="rgb(59, 130, 246)"
                      strokeWidth="0.5"
                      opacity="0.3"
                    />
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />
              </svg>
            </div>

            <div className="text-center text-white p-8 md:p-12 max-w-5xl mx-auto relative z-10">
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 mt-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent animate-slide-up">
                Discover Your Perfect
                <span className="block text-blue-400">Hairstyle</span>
              </h1>
              <p
                className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed text-gray-300 animate-slide-up"
                style={{ animationDelay: "0.1s" }}
              >
                Transform your look with AI-powered analysis. Get personalized
                hairstyle recommendations that perfectly complement your unique
                features and lifestyle.
              </p>
              <div
                className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-slide-up"
                style={{ animationDelay: "0.2s" }}
              >
                <Button
                  onClick={() =>
                    navigate("/upload", { state: { showGuide: true } })
                  }
                  variant="primary"
                  size="lg"
                  className="px-10 py-5 text-xl"
                >
                  Start Your Analysis
                </Button>
              </div>

              {/* How It Works Section - Moved to Hero */}
              <div
                className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center animate-slide-up"
                style={{ animationDelay: "0.3s" }}
              >
                <div className="text-center group">
                  <div className="relative mb-6">
                    <div className="bg-gradient-to-br from-purple-600 to-blue-600 rounded-full w-20 h-20 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-3xl">📸</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors duration-300">
                    Upload Photo
                  </h3>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Simply upload your photo to get started with our AI analysis
                  </p>
                </div>

                <div className="text-center group">
                  <div className="relative mb-6">
                    <div className="bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full w-20 h-20 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-blue-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-3xl">🤖</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors duration-300">
                    AI Analysis
                  </h3>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Our AI analyzes your face shape and features to find the
                    best match
                  </p>
                </div>

                <div className="text-center group">
                  <div className="relative mb-6">
                    <div className="bg-gradient-to-br from-indigo-600 to-purple-600 rounded-full w-20 h-20 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-indigo-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-3xl">✨</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-indigo-400 transition-colors duration-300">
                    Get Results
                  </h3>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Receive personalized hairstyle recommendations tailored for
                    you
                  </p>
                </div>
              </div>

              {/* Scroll Down Indicator */}
              <div className="mt-10 flex flex-col items-center animate-bounce hidden md:flex">
                <div className="text-gray-400 text-sm mb-2">
                  Scroll to explore
                </div>
                <div className="flex flex-col items-center">
                  <div className="w-6 h-10 border-2 border-gray-400 rounded-full flex items-start justify-center p-2">
                    <div className="w-1.5 h-2 bg-gray-400 rounded-full animate-scroll"></div>
                  </div>
                  <svg
                    className="w-6 h-6 text-gray-400 mt-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 14l-7 7m0 0l-7-7m7 7V3"
                    />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Discover Section */}
      <div className="bg-background pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <DiscoverSection />
        </div>
      </div>
    </div>
  );
};

export default Analyze;
