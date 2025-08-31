import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const Navbar = ({ transparent = true, showBackButton = false, backPath = '/', user = null, onLogout = null }) => {
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setIsScrolled(scrollTop > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showUserMenu && !event.target.closest('.user-menu')) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showUserMenu]);

  // Determine if navbar should use dark styling (better contrast)
  const useDarkStyling = !transparent || isScrolled;

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      useDarkStyling
        ? 'bg-gray-900/95 backdrop-blur-md shadow-lg border-b border-gray-700/50' 
        : 'bg-white/10 backdrop-blur-md border-b border-white/20'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side - Back button or Logo */}
          <div className="flex items-center">
            {showBackButton ? (
              <button
                type="button"
                onClick={() => navigate(backPath)}
                className={`p-2 rounded-full backdrop-blur-sm transition duration-300 ${
                  useDarkStyling
                    ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                    : 'text-white hover:text-gray-200 hover:bg-white/10'
                }`}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            ) : (
              <Link to="/" className={`text-2xl font-bold transition duration-300 ${
                useDarkStyling ? 'text-white' : 'text-white'
              }`}>
                HairMixer
              </Link>
            )}
          </div>

          {/* Center - Navigation Links */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <Link
                to="/"
                className={`px-3 py-2 rounded-md text-sm font-medium transition duration-300 ${
                  useDarkStyling
                    ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                    : 'text-white hover:text-gray-200 hover:bg-white/10'
                }`}
              >
                Home
              </Link>
              <Link
                to="/discover"
                className={`px-3 py-2 rounded-md text-sm font-medium transition duration-300 ${
                  useDarkStyling
                    ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                    : 'text-white hover:text-gray-200 hover:bg-white/10'
                }`}
              >
                Discover
              </Link>
              <Link
                to="/dashboard"
                className={`px-3 py-2 rounded-md text-sm font-medium transition duration-300 ${
                  useDarkStyling
                    ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                    : 'text-white hover:text-gray-200 hover:bg-white/10'
                }`}
              >
                Dashboard
              </Link>
            </div>
          </div>

          {/* Right side - Auth buttons or User info */}
          <div className="flex items-center space-x-4">
            {user ? (
              // Authenticated user content
              <div className="relative user-menu">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition duration-300 backdrop-blur-sm ${
                    useDarkStyling
                      ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                      : 'text-white hover:text-gray-200 hover:bg-white/10'
                  }`}
                >
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                    {user?.firstName?.charAt(0) || user?.email?.charAt(0) || 'U'}
                  </div>
                  <span>{user?.firstName || 'User'}</span>
                  <svg className={`w-4 h-4 transition-transform duration-200 ${showUserMenu ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {/* User Dropdown Menu */}
                {showUserMenu && (
                  <div className={`absolute right-0 mt-2 w-48 rounded-md shadow-lg backdrop-blur-md border ${
                    useDarkStyling
                      ? 'bg-gray-800/95 border-gray-600/50' 
                      : 'bg-white/95 border-white/20'
                  } z-50`}>
                    <div className="py-1">
                      <Link
                        to="/profile"
                        onClick={() => setShowUserMenu(false)}
                        className={`flex items-center px-4 py-2 text-sm transition duration-300 ${
                          useDarkStyling
                            ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                            : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100/50'
                        }`}
                      >
                        <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                        Profile Settings
                      </Link>
                      <div className={`border-t my-1 ${
                        useDarkStyling ? 'border-gray-600/50' : 'border-gray-200/50'
                      }`}></div>
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          onLogout();
                        }}
                        className={`flex items-center w-full px-4 py-2 text-sm transition duration-300 ${
                          useDarkStyling
                            ? 'text-red-400 hover:text-red-300 hover:bg-red-900/20' 
                            : 'text-red-600 hover:text-red-700 hover:bg-red-50/50'
                        }`}
                      >
                        <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                        Logout
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              // Guest user content
              <>
                <Link
                  to="/login"
                  className={`px-4 py-2 rounded-md text-sm font-medium transition duration-300 backdrop-blur-sm ${
                    useDarkStyling
                      ? 'text-gray-300 hover:text-white border border-gray-600/50 hover:bg-white/10' 
                      : 'text-white hover:text-gray-200 border border-white/50 hover:bg-white/20'
                  }`}
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className={`px-4 py-2 rounded-md text-sm font-medium transition duration-300 backdrop-blur-sm ${
                    useDarkStyling
                      ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700 border border-purple-500/30' 
                      : 'bg-white/20 text-white hover:bg-white/30 border border-white/30'
                  }`}
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className={`p-2 rounded-md backdrop-blur-sm transition duration-300 ${
                useDarkStyling
                  ? 'text-gray-300 hover:text-white hover:bg-white/10' 
                  : 'text-white hover:text-gray-200 hover:bg-white/10'
              } focus:outline-none`}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
