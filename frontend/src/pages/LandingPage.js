import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import AuthService from "../services/AuthService";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";

const LandingPage = () => {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        setUser(currentUser);
      } catch (error) {
        console.error("Authentication check failed:", error);
        setUser(null);
      }
    };

    checkAuth();
  }, []);

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      navigate("/");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <>
      <Navbar transparent={true} user={user} onLogout={handleLogout} />
      {/* Hero Section with Dark Theme */}
      <div className="min-h-screen bg-background pt-20 md:pt-24">
        <div className="bg-gradient-to-br from-background via-surface to-blue-900/20 min-h-screen overflow-hidden">
          <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
            {/* Dark geometric pattern background - Hidden on mobile to prevent overflow/clutter */}
            <div className="absolute inset-0 opacity-10 mt-20 pointer-events-none hidden md:block">
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
                    id="grid-landing"
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
                <rect width="100%" height="100%" fill="url(#grid-landing)" />
              </svg>
            </div>

            <div className="text-center text-white p-6 md:p-12 max-w-5xl mx-auto relative z-10 flex flex-col items-center justify-center min-h-[80vh] md:min-h-0">
              <h1
                className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent animate-slide-up leading-tight"
                style={{ animationDelay: "0.1s" }}
              >
                Transform Your Look with
                <span className="block text-blue-400 mt-2">HairMixer</span>
              </h1>
              <p
                className="text-lg md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed text-gray-300 animate-slide-up px-4"
                style={{ animationDelay: "0.2s" }}
              >
                Upload your photo, customize your preferences with our detailed
                wizard, and virtually try on hairstyles. Our system is trusted
                and validated by professional hairstylists and salons.
              </p>
              <div
                className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-slide-up w-full sm:w-auto px-4"
                style={{ animationDelay: "0.3s" }}
              >
                <Link to="/analyze" className="w-full sm:w-auto">
                  <Button
                    variant="primary"
                    size="lg"
                    className="w-full sm:w-auto text-lg px-8 py-4 md:py-5 md:px-10 shadow-lg shadow-primary/25"
                  >
                    Start Your Analysis
                  </Button>
                </Link>
                <Link to="/discover" className="w-full sm:w-auto">
                  <Button
                    variant="secondary"
                    size="lg"
                    className="w-full sm:w-auto text-lg px-8 py-4 md:py-5 md:px-10"
                  >
                    Explore Styles
                  </Button>
                </Link>
              </div>

              {/* How It Works - Compact Version */}
              <div
                className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 animate-slide-up w-full"
                style={{ animationDelay: "0.4s" }}
              >
                <Card
                  hover
                  className="p-6 group bg-surface/50 backdrop-blur-sm border-white/5"
                >
                  <div className="relative mb-4">
                    <div className="bg-gradient-to-br from-purple-600 to-blue-600 rounded-full w-14 h-14 md:w-16 md:h-16 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-2xl md:text-3xl">📸</span>
                    </div>
                  </div>
                  <h3 className="text-lg md:text-xl font-bold text-white mb-2">
                    1. Upload Photo
                  </h3>
                  <p className="text-sm md:text-base text-gray-300">
                    Our AI instantly analyzes your face shape and features
                  </p>
                </Card>

                <Card
                  hover
                  className="p-6 group bg-surface/50 backdrop-blur-sm border-white/5"
                >
                  <div className="relative mb-4">
                    <div className="bg-gradient-to-br from-blue-600 to-indigo-600 rounded-full w-14 h-14 md:w-16 md:h-16 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-blue-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-2xl md:text-3xl">⚙️</span>
                    </div>
                  </div>
                  <h3 className="text-lg md:text-xl font-bold text-white mb-2">
                    2. Customize
                  </h3>
                  <p className="text-sm md:text-base text-gray-300">
                    Refine results with our 11-step preference wizard
                  </p>
                </Card>

                <Card
                  hover
                  className="p-6 group bg-surface/50 backdrop-blur-sm border-white/5"
                >
                  <div className="relative mb-4">
                    <div className="bg-gradient-to-br from-indigo-600 to-purple-600 rounded-full w-14 h-14 md:w-16 md:h-16 flex items-center justify-center mx-auto shadow-lg group-hover:shadow-indigo-500/25 transition-all duration-300 group-hover:scale-110">
                      <span className="text-2xl md:text-3xl">✨</span>
                    </div>
                  </div>
                  <h3 className="text-lg md:text-xl font-bold text-white mb-2">
                    3. Virtual Try-On
                  </h3>
                  <p className="text-sm md:text-base text-gray-300">
                    See hairstyles directly on your photo with AI
                  </p>
                </Card>
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

      {/* Success Stories & Social Proof Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-background">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 md:mb-6">
              Trusted and Validated by Hairstylists and Salons
            </h2>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">
              See what our users are saying about their HairMixer experience
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 mb-12 md:mb-16">
            <Card hover className="p-6 group">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                  S
                </div>
                <div className="ml-4">
                  <h4 className="text-white font-semibold text-base md:text-lg">
                    Sarah M.
                  </h4>
                  <div className="flex text-yellow-400 text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base leading-relaxed">
                "Finally found the perfect cut for my face shape! The AI
                recommendations were spot-on and my stylist was impressed."
              </p>
            </Card>

            <Card hover className="p-6 group">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                  M
                </div>
                <div className="ml-4">
                  <h4 className="text-white font-semibold text-base md:text-lg">
                    Marcus T.
                  </h4>
                  <div className="flex text-yellow-400 text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base leading-relaxed">
                "Super easy to use and gave me confidence to try a new style.
                The results were exactly what I was looking for!"
              </p>
            </Card>

            <Card hover className="p-6 group">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center text-white font-bold text-lg flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                  A
                </div>
                <div className="ml-4">
                  <h4 className="text-white font-semibold text-base md:text-lg">
                    Aisha K.
                  </h4>
                  <div className="flex text-yellow-400 text-base">★★★★★</div>
                </div>
              </div>
              <p className="text-gray-300 italic text-sm md:text-base leading-relaxed">
                "Love how it considers my lifestyle! Perfect recommendations for
                both work and weekend looks."
              </p>
            </Card>
          </div>
        </div>
      </div>

      {/* Features & Benefits Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-surface">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 md:mb-6">
              Why Choose HairMixer?
            </h2>
            <p className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto">
              Powered by cutting-edge AI technology and trusted and validated by
              professional hairstylists and salons worldwide
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 md:gap-12 items-center mb-12 md:mb-16">
            <div>
              <div className="space-y-6">
                <Card hover className="p-6 flex items-start space-x-4 group">
                  <div className="bg-purple-500/20 rounded-lg p-3 flex-shrink-0 border border-purple-500/30 group-hover:scale-110 transition-transform duration-300">
                    <span className="text-2xl">🎯</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors duration-300">
                      Precision Face Analysis
                    </h3>
                    <p className="text-base text-gray-300 leading-relaxed">
                      Advanced AI technology analyzes facial landmarks to
                      determine your unique face shape with professional
                      accuracy.
                    </p>
                  </div>
                </Card>

                <Card hover className="p-6 flex items-start space-x-4 group">
                  <div className="bg-blue-500/20 rounded-lg p-3 flex-shrink-0 border border-blue-500/30 group-hover:scale-110 transition-transform duration-300">
                    <span className="text-2xl">👥</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors duration-300">
                      Detailed Preference Wizard
                    </h3>
                    <p className="text-base text-gray-300 leading-relaxed">
                      Our 11-step wizard captures your hair type, lifestyle, and
                      maintenance preferences for truly personalized results.
                    </p>
                  </div>
                </Card>

                <Card hover className="p-6 flex items-start space-x-4 group">
                  <div className="bg-green-500/20 rounded-lg p-3 flex-shrink-0 border border-green-500/30 group-hover:scale-110 transition-transform duration-300">
                    <span className="text-2xl">💎</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2 group-hover:text-green-400 transition-colors duration-300">
                      Virtual Try-On
                    </h3>
                    <p className="text-base text-gray-300 leading-relaxed">
                      Visualize your new look before you commit. Our AI overlays
                      hairstyles directly onto your photo for a realistic
                      preview.
                    </p>
                  </div>
                </Card>
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 backdrop-blur-sm rounded-2xl p-6 md:p-10 border border-purple-500/20">
              <h3 className="text-2xl font-bold text-white mb-8 text-center">
                Our Impact
              </h3>
              <div className="grid grid-cols-2 gap-4 md:gap-6">
                <div className="text-center bg-white/5 rounded-lg p-4 md:p-6 border border-white/10">
                  <div className="text-2xl md:text-4xl font-bold text-purple-400 mb-2">
                    50+
                  </div>
                  <div className="text-xs md:text-sm text-gray-300">
                    Happy Users
                  </div>
                </div>
                <div className="text-center bg-white/5 rounded-lg p-4 md:p-6 border border-white/10">
                  <div className="text-2xl md:text-4xl font-bold text-blue-400 mb-2">
                    55+
                  </div>
                  <div className="text-xs md:text-sm text-gray-300">
                    Hairstyles
                  </div>
                </div>
                <div className="text-center bg-white/5 rounded-lg p-4 md:p-6 border border-white/10">
                  <div className="text-2xl md:text-4xl font-bold text-indigo-400 mb-2">
                    94%
                  </div>
                  <div className="text-xs md:text-sm text-gray-300">
                    Satisfaction
                  </div>
                </div>
                <div className="text-center bg-white/5 rounded-lg p-4 md:p-6 border border-white/10">
                  <div className="text-2xl md:text-4xl font-bold text-green-400 mb-2">
                    2min
                  </div>
                  <div className="text-xs md:text-sm text-gray-300">
                    Average Time
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="py-12 md:py-20 px-4 sm:px-6 lg:px-8 bg-background">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 md:mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 md:mb-6">
              Frequently Asked Questions
            </h2>
            <p className="text-lg md:text-xl text-gray-300">
              Everything you need to know about HairMixer
            </p>
          </div>

          <div className="space-y-4">
            <Card hover className="p-6 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-purple-400 transition-colors duration-300">
                How accurate are the hairstyle recommendations?
              </h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">
                Our AI technology has a 99% accuracy rate in face shape
                detection. We combine this with your detailed preferences from
                our 11-step wizard to suggest styles that truly suit your
                features and lifestyle.
              </p>
            </Card>

            <Card hover className="p-6 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors duration-300">
                Is my photo data secure and private?
              </h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">
                Absolutely! Your photos are processed securely and are not
                stored permanently on our servers unless you choose to save
                specific results. We use enterprise-grade encryption and follow
                strict privacy protocols.
              </p>
            </Card>

            <Card hover className="p-6 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-indigo-400 transition-colors duration-300">
                Can I see what the hairstyle looks like on me?
              </h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">
                Yes! Our Virtual Try-On feature uses advanced generative AI to
                overlay the recommended hairstyles onto your uploaded photo,
                giving you a realistic preview of your new look.
              </p>
            </Card>

            <Card hover className="p-6 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-green-400 transition-colors duration-300">
                What type of photo works best?
              </h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">
                For best results, use a clear, front-facing photo with good
                lighting. Avoid sunglasses, hats, or anything covering your
                face. Natural lighting and a neutral expression work perfectly.
              </p>
            </Card>

            <Card hover className="p-6 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-purple-400 transition-colors duration-300">
                How many hairstyle options will I get?
              </h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">
                You'll typically receive 5-8 personalized recommendations based
                on your face shape and preferences. Each recommendation includes
                styling tips, maintenance level, and suitability for different
                occasions.
              </p>
            </Card>

            <Card hover className="p-6 group">
              <h3 className="text-lg md:text-xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors duration-300">
                Can I use HairMixer for special occasions?
              </h3>
              <p className="text-sm md:text-base text-gray-300 leading-relaxed">
                Yes! Our preference system allows you to specify occasions like
                weddings, work, casual outings, or formal events. We'll tailor
                recommendations to match the styling needs for each occasion.
              </p>
            </Card>
          </div>
        </div>
      </div>
    </>
  );
};

export default LandingPage;
