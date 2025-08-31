import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthService from '../services/AuthService';
import DiscoverSection from '../components/DiscoverSection';
import Navbar from '../components/Navbar';

const Dashboard = () => {
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
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600">
      {/* Navigation */}
      <Navbar 
        transparent={true} 
        user={user} 
        onLogout={handleLogout} 
      />

      {/* Full Screen Hero Section */}
      <div className="h-screen">
        <div className="bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 h-full overflow-hidden">
          <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
            {/* Dark geometric pattern background */}
            <div className="absolute inset-0 opacity-10">
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
              <svg className="absolute inset-0 w-full h-full" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
                    <path d="M 60 0 L 0 0 0 60" fill="none" stroke="rgb(59, 130, 246)" strokeWidth="0.5" opacity="0.3"/>
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)" />
              </svg>
            </div>
            
            <div className="text-center text-white p-8 md:p-12 max-w-5xl mx-auto relative z-10">
              <div className="mb-6">
                <span className="inline-block bg-blue-500/20 text-blue-300 px-4 py-2 rounded-full text-sm font-medium border border-blue-500/30 backdrop-blur-sm">
                  Welcome back, {user?.firstName || 'User'}
                </span>
              </div>
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent">
                Discover Your Perfect
                <span className="block text-blue-400">Hairstyle</span>
              </h1>
              <p className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed text-gray-300">
                Transform your look with AI-powered analysis. Get personalized hairstyle recommendations that perfectly complement your unique features and lifestyle.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <button
                  onClick={() => navigate('/upload')}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-4 px-8 md:py-5 md:px-10 rounded-lg text-lg md:text-xl transition duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-2xl border border-blue-500/30"
                >
                  Start Your Analysis
                </button>
                <button
                  onClick={() => console.log('View trends clicked')}
                  className="bg-white/10 hover:bg-white/20 text-white font-semibold py-4 px-8 md:py-5 md:px-10 rounded-lg text-lg md:text-xl transition duration-300 ease-in-out backdrop-blur-sm border border-white/20 hover:border-white/40"
                >
                  Explore Trends
                </button>
              </div>
              
              {/* Stats or features */}
              <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-blue-400 mb-2">10K+</div>
                  <div className="text-gray-300">Happy Users</div>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-purple-400 mb-2">500+</div>
                  <div className="text-gray-300">Hairstyles</div>
                </div>
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-indigo-400 mb-2">98%</div>
                  <div className="text-gray-300">Satisfaction</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content Sections */}
      <div className="bg-gray-900 py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">

          {/* How It Works Section */}
          <div className="mb-20">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">How It Works</h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">Get personalized hairstyle recommendations in three simple steps</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
              <div className="text-center group">
                <div className="relative mb-8">
                  <div className="bg-gradient-to-br from-purple-600 to-blue-600 rounded-full w-24 h-24 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300 group-hover:scale-110">
                    <span className="text-4xl">📸</span>
                  </div>
                  <div className="absolute -inset-1 bg-gradient-to-br from-purple-600 to-blue-600 rounded-full opacity-20 blur-sm group-hover:opacity-40 transition-opacity duration-300"></div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-purple-400 transition-colors duration-300">Upload Photo</h3>
                <p className="text-gray-300 text-lg leading-relaxed">Simply upload your photo to get started with our AI analysis</p>
              </div>
              
              <div className="text-center group">
                <div className="relative mb-8">
                  <div className="bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full w-24 h-24 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-blue-500/25 transition-all duration-300 group-hover:scale-110">
                    <span className="text-4xl">🤖</span>
                  </div>
                  <div className="absolute -inset-1 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full opacity-20 blur-sm group-hover:opacity-40 transition-opacity duration-300"></div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-blue-400 transition-colors duration-300">AI Analysis</h3>
                <p className="text-gray-300 text-lg leading-relaxed">Our AI analyzes your face shape and features to find the best match</p>
              </div>
              
              <div className="text-center group">
                <div className="relative mb-8">
                  <div className="bg-gradient-to-br from-indigo-600 to-purple-600 rounded-full w-24 h-24 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-indigo-500/25 transition-all duration-300 group-hover:scale-110">
                    <span className="text-4xl">✨</span>
                  </div>
                  <div className="absolute -inset-1 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-full opacity-20 blur-sm group-hover:opacity-40 transition-opacity duration-300"></div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-indigo-400 transition-colors duration-300">Get Results</h3>
                <p className="text-gray-300 text-lg leading-relaxed">Receive personalized hairstyle recommendations tailored for you</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Discover Section */}
      <div className="bg-gray-900 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <DiscoverSection />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;