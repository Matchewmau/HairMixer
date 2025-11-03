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
      {/* Hero Section with Dark Theme */}
      <div className="min-h-screen bg-gray-900 pt-20 md:pt-24">
        <div className="bg-gradient-to-br from-gray-900 via-slate-800 to-blue-900 min-h-screen overflow-hidden">
          <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
            {/* Dark geometric pattern background */}
            <div className="absolute inset-0 opacity-10 mt-20">
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
                  <pattern id="grid-landing" width="60" height="60" patternUnits="userSpaceOnUse">
                    <path d="M 60 0 L 0 0 0 60" fill="none" stroke="rgb(59, 130, 246)" strokeWidth="0.5" opacity="0.3"/>
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid-landing)" />
              </svg>
            </div>
            
            <div className="text-center text-white p-8 md:p-12 max-w-5xl mx-auto relative z-10">
              <div className="mb-6">
                <span className="inline-block bg-blue-500/20 text-blue-300 px-4 py-2 rounded-full text-sm font-medium border border-blue-500/30 backdrop-blur-sm">
                  AI-Powered Hair Analysis
                </span>
              </div>
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent">
                Transform Your Look with
                <span className="block text-blue-400">HairMixer</span>
              </h1>
              <p className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed text-gray-300">
                Upload your photo and discover the perfect hairstyle for you. Powered by advanced AI technology trusted by thousands worldwide.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <Link
                  to="/upload"
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-4 px-8 md:py-5 md:px-10 rounded-lg text-lg md:text-xl transition duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-2xl border border-blue-500/30"
                >
                  Start Your Analysis
                </Link>
                <Link
                  to="/discover"
                  className="bg-white/10 hover:bg-white/20 text-white font-semibold py-4 px-8 md:py-5 md:px-10 rounded-lg text-lg md:text-xl transition duration-300 ease-in-out backdrop-blur-sm border border-white/20 hover:border-white/40"
                >
                  Explore Styles
                </Link>
              </div>      
              {/* How It Works - Compact Version */}
              <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 group hover:border-purple-500/30 transition-all duration-300">
                  <div className="relative mb-4">
                    <div className="bg-gradient-to-br from-purple-600 to-blue-600 rounded-full w-16 h-16 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-3xl">📸</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">Upload Photo</h3>
                  <p className="text-gray-300">Simply upload your photo to get started</p>
                </div>
                
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 group hover:border-blue-500/30 transition-all duration-300">
                  <div className="relative mb-4">
                    <div className="bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full w-16 h-16 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-blue-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-3xl">🤖</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">AI Analysis</h3>
                  <p className="text-gray-300">Our AI analyzes your face shape and features</p>
                </div>
                
                <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 group hover:border-indigo-500/30 transition-all duration-300">
                  <div className="relative mb-4">
                    <div className="bg-gradient-to-br from-indigo-600 to-purple-600 rounded-full w-16 h-16 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-indigo-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-3xl">✨</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">Get Results</h3>
                  <p className="text-gray-300">Receive personalized hairstyle recommendations</p>
                </div>
              </div>

              {/* Scroll Down Indicator */}
              <div className="mt-4 flex flex-col items-center animate-bounce">
                <div className="text-gray-400 text-sm mb-2">Scroll to explore</div>
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

      {/* Success Stories & Social Proof Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-gray-900">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 md:mb-6">Trusted by Thousands</h2>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">See what our users are saying about their HairMixer experience</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 mb-12 md:mb-16">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-purple-500/30 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300 group">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0 group-hover:scale-110 transition-transform duration-300">S</div>
                <div className="ml-4">
                  <h4 className="text-white font-semibold text-base md:text-lg">Sarah M.</h4>
                  <div className="flex text-yellow-400 text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base leading-relaxed">"Finally found the perfect cut for my face shape! The AI recommendations were spot-on and my stylist was impressed."</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-blue-500/30 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300 group">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0 group-hover:scale-110 transition-transform duration-300">M</div>
                <div className="ml-4">
                  <h4 className="text-white font-semibold text-base md:text-lg">Marcus T.</h4>
                  <div className="flex text-yellow-400 text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base leading-relaxed">"Super easy to use and gave me confidence to try a new style. The results were exactly what I was looking for!"</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-green-500/30 hover:shadow-lg hover:shadow-green-500/10 transition-all duration-300 group">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0 group-hover:scale-110 transition-transform duration-300">A</div>
                <div className="ml-4">
                  <h4 className="text-white font-semibold text-base md:text-lg">Aisha K.</h4>
                  <div className="flex text-yellow-400 text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base leading-relaxed">"Love how it considers my lifestyle! Perfect recommendations for both work and weekend looks."</p>
            </div>
          </div>

          <div className="text-center">
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl px-8 py-8 border border-white/10">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 divide-y sm:divide-y-0 sm:divide-x divide-white/10">
                <div className="text-center pt-6 sm:pt-0">
                  <div className="text-2xl md:text-3xl font-bold text-white mb-1">4.9/5</div>
                  <div className="text-sm md:text-base text-gray-400">Average Rating</div>
                  <div className="flex justify-center text-yellow-400 mt-2 text-base">★★★★★</div>
                </div>
                <div className="text-center pt-6 sm:pt-0">
                  <div className="text-2xl md:text-3xl font-bold text-blue-400 mb-1">10K+</div>
                  <div className="text-sm md:text-base text-gray-400">Happy Users</div>
                </div>
                <div className="text-center pt-6 sm:pt-0">
                  <div className="text-2xl md:text-3xl font-bold text-purple-400 mb-1">50K+</div>
                  <div className="text-sm md:text-base text-gray-400">Photos Analyzed</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features & Benefits Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-slate-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 md:mb-6">Why Choose HairMixer?</h2>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">Powered by cutting-edge AI technology and trusted by thousands of users worldwide</p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 md:gap-12 items-center mb-12 md:mb-16">
            <div>
              <div className="space-y-6">
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-purple-500/30 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300 group">
                  <div className="flex items-start space-x-4">
                    <div className="bg-purple-500/20 rounded-lg p-3 flex-shrink-0 border border-purple-500/30 group-hover:scale-110 transition-transform duration-300">
                      <span className="text-2xl">🎯</span>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors duration-300">Precision Face Analysis</h3>
                      <p className="text-base text-gray-300 leading-relaxed">Advanced AI technology analyzes 68+ facial landmarks to determine your unique face shape with professional accuracy.</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-blue-500/30 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300 group">
                  <div className="flex items-start space-x-4">
                    <div className="bg-blue-500/20 rounded-lg p-3 flex-shrink-0 border border-blue-500/30 group-hover:scale-110 transition-transform duration-300">
                      <span className="text-2xl">👥</span>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors duration-300">Lifestyle-Based Recommendations</h3>
                      <p className="text-base text-gray-300 leading-relaxed">Get suggestions tailored to your daily routine, maintenance preferences, and special occasions.</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-green-500/30 hover:shadow-lg hover:shadow-green-500/10 transition-all duration-300 group">
                  <div className="flex items-start space-x-4">
                    <div className="bg-green-500/20 rounded-lg p-3 flex-shrink-0 border border-green-500/30 group-hover:scale-110 transition-transform duration-300">
                      <span className="text-2xl">💎</span>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-green-400 transition-colors duration-300">Professional Quality</h3>
                      <p className="text-base text-gray-300 leading-relaxed">Recommendations validated by professional stylists and based on proven beauty principles.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 backdrop-blur-sm rounded-2xl p-8 md:p-10 border border-purple-500/20">
              <h3 className="text-2xl font-bold text-white mb-8 text-center">Our Impact</h3>
              <div className="grid grid-cols-2 gap-6">
                <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-purple-400 mb-2">10K+</div>
                  <div className="text-sm text-gray-300">Happy Users</div>
                </div>
                <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-blue-400 mb-2">500+</div>
                  <div className="text-sm text-gray-300">Hairstyles</div>
                </div>
                <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-indigo-400 mb-2">98%</div>
                  <div className="text-sm text-gray-300">Satisfaction</div>
                </div>
                <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                  <div className="text-3xl md:text-4xl font-bold text-green-400 mb-2">2min</div>
                  <div className="text-sm text-gray-300">Average Time</div>
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
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 md:mb-6">Frequently Asked Questions</h2>
            <p className="text-lg md:text-xl text-gray-300">Everything you need to know about HairMixer</p>
          </div>
          
          <div className="space-y-4">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-purple-500/30 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-purple-400 transition-colors duration-300">How accurate are the hairstyle recommendations?</h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">Our AI technology has a 99% accuracy rate in face shape detection and our recommendations are validated by professional stylists. The system analyzes over 68 facial landmarks to ensure precise results.</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-blue-500/30 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors duration-300">Is my photo data secure and private?</h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">Absolutely! Your photos are processed securely and are not stored permanently on our servers. We use enterprise-grade encryption and follow strict privacy protocols. You can delete your data at any time.</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-indigo-500/30 hover:shadow-lg hover:shadow-indigo-500/10 transition-all duration-300 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-indigo-400 transition-colors duration-300">Do I need to create an account to use HairMixer?</h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">No account required for basic recommendations! You can upload a photo and get instant results. Creating an account allows you to save your preferences and access your recommendation history.</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-green-500/30 hover:shadow-lg hover:shadow-green-500/10 transition-all duration-300 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-green-400 transition-colors duration-300">What type of photo works best?</h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">For best results, use a clear, front-facing photo with good lighting. Avoid sunglasses, hats, or anything covering your face. Natural lighting and a neutral expression work perfectly.</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-purple-500/30 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-purple-400 transition-colors duration-300">How many hairstyle options will I get?</h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">You'll typically receive 5-8 personalized recommendations based on your face shape and preferences. Each recommendation includes styling tips, maintenance level, and suitability for different occasions.</p>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-blue-500/30 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors duration-300">Can I use HairMixer for special occasions?</h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">Yes! Our preference system allows you to specify occasions like weddings, work, casual outings, or formal events. We'll tailor recommendations to match the styling needs for each occasion.</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default LandingPage;