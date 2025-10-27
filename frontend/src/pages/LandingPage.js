import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AuthService from '../services/AuthService';

const LandingPage = () => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error('Authentication check failed:', error);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate('/');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <>
      <Navbar 
        transparent={true} 
        user={user} 
        onLogout={handleLogout} 
      />
      <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center relative pt-20 md:pt-0">
        <div className="max-w-4xl mx-auto px-4 py-8 md:py-0 text-center">
        <div className="bg-white rounded-lg shadow-2xl p-6 md:p-12">
          <h1 className="text-3xl md:text-6xl font-bold text-gray-800 mb-4 md:mb-6">
            Hair<span className="text-purple-600">Mixer</span>
          </h1>
          <p className="text-base md:text-xl text-gray-600 mb-6 md:mb-8 max-w-2xl mx-auto">
            Transform your look with AI-powered hairstyle recommendations. 
            Upload your photo and discover the perfect hairstyle for you.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-6 md:mb-8">
            <Link 
              to="/upload"
              className="w-full sm:w-auto bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-8 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
            >
              Start Analyzing
            </Link>
          </div>
          
          <div className="mt-8 md:mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-12 h-12 md:w-16 md:h-16 flex items-center justify-center mx-auto mb-3 md:mb-4">
                <span className="text-xl md:text-2xl">📸</span>
              </div>
              <h3 className="text-base md:text-lg font-semibold text-gray-800 mb-2">Upload Photo</h3>
              <p className="text-sm md:text-base text-gray-600">Simply upload your photo to get started</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-12 h-12 md:w-16 md:h-16 flex items-center justify-center mx-auto mb-3 md:mb-4">
                <span className="text-xl md:text-2xl">🤖</span>
              </div>
              <h3 className="text-base md:text-lg font-semibold text-gray-800 mb-2">AI Analysis</h3>
              <p className="text-sm md:text-base text-gray-600">Our AI analyzes your face shape and features</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-12 h-12 md:w-16 md:h-16 flex items-center justify-center mx-auto mb-3 md:mb-4">
                <span className="text-xl md:text-2xl">✨</span>
              </div>
              <h3 className="text-base md:text-lg font-semibold text-gray-800 mb-2">Get Results</h3>
              <p className="text-sm md:text-base text-gray-600">Receive personalized hairstyle recommendations</p>
            </div>
          </div>
        </div>
        </div>
        
        {/* Scroll Indicator - Hidden on mobile */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce hidden md:flex">
          <div className="flex flex-col items-center text-white">
            <span className="text-sm font-medium mb-2 opacity-80">Discover More</span>
            <div className="w-6 h-10 border-2 border-white rounded-full flex justify-center opacity-80">
              <div className="w-1 h-3 bg-white rounded-full mt-2 animate-pulse"></div>
            </div>
            <svg className="w-4 h-4 mt-2 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </div>
      </div>

      {/* Smooth Gradient Transition */}
      <div className="h-20 bg-gradient-to-b from-blue-600 to-slate-800"></div>

      {/* Success Stories & Social Proof Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-slate-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-3 md:mb-4">Trusted by Thousands</h2>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">See what our users are saying about their HairMixer experience</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 mb-12 md:mb-16">
            <div className="bg-slate-700/50 backdrop-blur-sm rounded-xl p-4 md:p-6 border border-slate-600/30 hover:border-purple-500/30 transition-all duration-300">
              <div className="flex items-center mb-3 md:mb-4">
                <div className="w-10 h-10 md:w-12 md:h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-base md:text-lg flex-shrink-0">S</div>
                <div className="ml-3">
                  <h4 className="text-white font-semibold text-sm md:text-base">Sarah M.</h4>
                  <div className="flex text-yellow-400 text-sm md:text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base">"Finally found the perfect cut for my face shape! The AI recommendations were spot-on and my stylist was impressed."</p>
            </div>
            
            <div className="bg-slate-700/50 backdrop-blur-sm rounded-xl p-4 md:p-6 border border-slate-600/30 hover:border-blue-500/30 transition-all duration-300">
              <div className="flex items-center mb-3 md:mb-4">
                <div className="w-10 h-10 md:w-12 md:h-12 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-full flex items-center justify-center text-white font-bold text-base md:text-lg flex-shrink-0">M</div>
                <div className="ml-3">
                  <h4 className="text-white font-semibold text-sm md:text-base">Marcus T.</h4>
                  <div className="flex text-yellow-400 text-sm md:text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base">"Super easy to use and gave me confidence to try a new style. The results were exactly what I was looking for!"</p>
            </div>
            
            <div className="bg-slate-700/50 backdrop-blur-sm rounded-xl p-4 md:p-6 border border-slate-600/30 hover:border-green-500/30 transition-all duration-300">
              <div className="flex items-center mb-3 md:mb-4">
                <div className="w-10 h-10 md:w-12 md:h-12 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center text-white font-bold text-base md:text-lg flex-shrink-0">A</div>
                <div className="ml-3">
                  <h4 className="text-white font-semibold text-sm md:text-base">Aisha K.</h4>
                  <div className="flex text-yellow-400 text-sm md:text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base">"Love how it considers my lifestyle! Perfect recommendations for both work and weekend looks."</p>
            </div>
          </div>

          <div className="text-center">
            <div className="bg-slate-700/30 backdrop-blur-sm rounded-2xl px-6 md:px-8 py-4 md:py-6 border border-slate-600/20">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 divide-y sm:divide-y-0 sm:divide-x divide-slate-600">
                <div className="text-center pt-4 sm:pt-0">
                  <div className="text-xl md:text-2xl font-bold text-white">4.9/5</div>
                  <div className="text-xs md:text-sm text-gray-300">Average Rating</div>
                  <div className="flex justify-center text-yellow-400 mt-1 text-sm md:text-base">★★★★★</div>
                </div>
                <div className="text-center pt-4 sm:pt-0">
                  <div className="text-xl md:text-2xl font-bold text-white">10K+</div>
                  <div className="text-xs md:text-sm text-gray-300">Happy Users</div>
                </div>
                <div className="text-center pt-4 sm:pt-0">
                  <div className="text-xl md:text-2xl font-bold text-white">50K+</div>
                  <div className="text-xs md:text-sm text-gray-300">Photos Analyzed</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features & Benefits Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-gray-900">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-3 md:mb-4">Why Choose HairMixer?</h2>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">Powered by cutting-edge AI technology and trusted by thousands of users worldwide</p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 md:gap-12 items-center mb-12 md:mb-16">
            <div>
              <div className="space-y-6 md:space-y-8">
                <div className="flex items-start space-x-3 md:space-x-4">
                  <div className="bg-purple-900/30 backdrop-blur-sm rounded-lg p-2 md:p-3 flex-shrink-0 border border-purple-500/20">
                    <span className="text-xl md:text-2xl">🎯</span>
                  </div>
                  <div>
                    <h3 className="text-lg md:text-xl font-bold text-white mb-2">Precision Face Analysis</h3>
                    <p className="text-sm md:text-base text-gray-300">Advanced AI technology analyzes 68+ facial landmarks to determine your unique face shape with professional accuracy.</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3 md:space-x-4">
                  <div className="bg-blue-900/30 backdrop-blur-sm rounded-lg p-2 md:p-3 flex-shrink-0 border border-blue-500/20">
                    <span className="text-xl md:text-2xl">👥</span>
                  </div>
                  <div>
                    <h3 className="text-lg md:text-xl font-bold text-white mb-2">Lifestyle-Based Recommendations</h3>
                    <p className="text-sm md:text-base text-gray-300">Get suggestions tailored to your daily routine, maintenance preferences, and special occasions.</p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3 md:space-x-4">
                  <div className="bg-green-900/30 backdrop-blur-sm rounded-lg p-2 md:p-3 flex-shrink-0 border border-green-500/20">
                    <span className="text-xl md:text-2xl">💎</span>
                  </div>
                  <div>
                    <h3 className="text-lg md:text-xl font-bold text-white mb-2">Professional Quality</h3>
                    <p className="text-sm md:text-base text-gray-300">Recommendations validated by professional stylists and based on proven beauty principles.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 backdrop-blur-sm rounded-2xl p-6 md:p-8 border border-purple-500/20">
              <div className="grid grid-cols-2 gap-4 md:gap-6">
                <div className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-purple-400 mb-2">10K+</div>
                  <div className="text-xs md:text-base text-gray-300">Happy Users</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-blue-400 mb-2">500+</div>
                  <div className="text-xs md:text-base text-gray-300">Hairstyles</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-indigo-400 mb-2">98%</div>
                  <div className="text-xs md:text-base text-gray-300">Satisfaction</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-green-400 mb-2">2min</div>
                  <div className="text-xs md:text-base text-gray-300">Average Time</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-gray-900">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-3 md:mb-4">Frequently Asked Questions</h2>
            <p className="text-lg md:text-xl text-gray-300">Everything you need to know about HairMixer</p>
          </div>
          
          <div className="space-y-6 md:space-y-8">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg shadow-md p-4 md:p-6 hover:shadow-purple-500/10 hover:bg-slate-800/70 transition-all duration-300 border border-slate-700/50">
              <h3 className="text-lg md:text-xl font-bold text-white mb-2 md:mb-3">How accurate are the hairstyle recommendations?</h3>
              <p className="text-sm md:text-base text-gray-300">Our AI technology has a 99% accuracy rate in face shape detection and our recommendations are validated by professional stylists. The system analyzes over 68 facial landmarks to ensure precise results.</p>
            </div>
            
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg shadow-md p-4 md:p-6 hover:shadow-purple-500/10 hover:bg-slate-800/70 transition-all duration-300 border border-slate-700/50">
              <h3 className="text-lg md:text-xl font-bold text-white mb-2 md:mb-3">Is my photo data secure and private?</h3>
              <p className="text-sm md:text-base text-gray-300">Absolutely! Your photos are processed securely and are not stored permanently on our servers. We use enterprise-grade encryption and follow strict privacy protocols. You can delete your data at any time.</p>
            </div>
            
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg shadow-md p-4 md:p-6 hover:shadow-purple-500/10 hover:bg-slate-800/70 transition-all duration-300 border border-slate-700/50">
              <h3 className="text-lg md:text-xl font-bold text-white mb-2 md:mb-3">Do I need to create an account to use HairMixer?</h3>
              <p className="text-sm md:text-base text-gray-300">No account required for basic recommendations! You can upload a photo and get instant results. Creating an account allows you to save your preferences and access your recommendation history.</p>
            </div>
            
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg shadow-md p-4 md:p-6 hover:shadow-purple-500/10 hover:bg-slate-800/70 transition-all duration-300 border border-slate-700/50">
              <h3 className="text-lg md:text-xl font-bold text-white mb-2 md:mb-3">What type of photo works best?</h3>
              <p className="text-sm md:text-base text-gray-300">For best results, use a clear, front-facing photo with good lighting. Avoid sunglasses, hats, or anything covering your face. Natural lighting and a neutral expression work perfectly.</p>
            </div>
            
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg shadow-md p-4 md:p-6 hover:shadow-purple-500/10 hover:bg-slate-800/70 transition-all duration-300 border border-slate-700/50">
              <h3 className="text-lg md:text-xl font-bold text-white mb-2 md:mb-3">How many hairstyle options will I get?</h3>
              <p className="text-sm md:text-base text-gray-300">You'll typically receive 5-8 personalized recommendations based on your face shape and preferences. Each recommendation includes styling tips, maintenance level, and suitability for different occasions.</p>
            </div>
            
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg shadow-md p-4 md:p-6 hover:shadow-purple-500/10 hover:bg-slate-800/70 transition-all duration-300 border border-slate-700/50">
              <h3 className="text-lg md:text-xl font-bold text-white mb-2 md:mb-3">Can I use HairMixer for special occasions?</h3>
              <p className="text-sm md:text-base text-gray-300">Yes! Our preference system allows you to specify occasions like weddings, work, casual outings, or formal events. We'll tailor recommendations to match the styling needs for each occasion.</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default LandingPage;