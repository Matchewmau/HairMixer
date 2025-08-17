import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthService from '../services/AuthService';
import DiscoverSection from '../components/DiscoverSection';

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [imageError, setImageError] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
        } else {
          navigate('/login');
        }
      } catch (error) {
        console.error('Authentication check failed:', error);
        navigate('/login');
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const handleStartAnalyzing = () => {
    navigate('/upload');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">
                Hair<span className="text-blue-600">Mixer</span>
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-gray-700">Welcome, {user?.firstName || 'User'}!</span>
              <button
                onClick={handleLogout}
                className="bg-red-500 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm font-medium transition duration-200"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative w-full h-96 md:h-[500px] lg:h-[600px] overflow-hidden">
        {/* Hero Background Image */}
        {!imageError ? (
          <img
            src="/dashboard/heroImg.png"
            alt="HairMixer Hero"
            className="absolute top-0 left-1/2 w-4/6 h-full object-cover -translate-x-1/2"
            onError={() => setImageError(true)}
          />
        ) : (
          // Fallback gradient background
          <div className="absolute inset-0 bg-gradient-to-br from-purple-300 via-blue-500 to-blue-700"></div>
        )}
        
        {/* Overlay for better text readability */}
        <div className="absolute inset-0 bg-black bg-opacity-20"></div>
        
        {/* Hero Content */}
        <div className="relative z-10 h-full flex flex-col justify-end items-center text-center p-8">
          <div className="mb-8">
            {/* System Name */}
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6 drop-shadow-lg">
              Hair<span className="text-blue-500">Mixer</span>
            </h1>
            
            {/* Quote */}
            <p className="text-lg md:text-xl lg:text-2xl text-white mb-8 max-w-4xl mx-auto leading-relaxed drop-shadow-md">
              Find Your Perfect Hairstyle: HairMixer analyzes your face shape to recommend personalized styles that fit you flawlessly.
            </p>
            
            {/* Start Analyzing Button */}
            <button
              onClick={handleStartAnalyzing}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-8 md:py-5 md:px-10 rounded-lg text-lg md:text-xl transition duration-300 ease-in-out transform hover:scale-105 shadow-xl hover:shadow-2xl"
            >
              Start Analyzing
            </button>
          </div>
        </div>
      </div>

      {/* Discover Section */}
      <DiscoverSection />

      {/* How It Works Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900 mb-12">How It Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">📸</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Upload Photo</h3>
                <p className="text-gray-600">Simply upload your photo to get started with our AI analysis</p>
              </div>
              <div className="text-center">
                <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">🤖</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">AI Analysis</h3>
                <p className="text-gray-600">Our AI analyzes your face shape and features to find the best match</p>
              </div>
              <div className="text-center">
                <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">✨</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Get Results</h3>
                <p className="text-gray-600">Receive personalized hairstyle recommendations tailored for you</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Dashboard;