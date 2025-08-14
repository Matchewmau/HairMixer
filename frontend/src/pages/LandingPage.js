import React from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center">
      <div className="max-w-4xl mx-auto px-4 text-center">
        <div className="bg-white rounded-lg shadow-2xl p-8 md:p-12">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-800 mb-6">
            Hair<span className="text-purple-600">Mixer</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Transform your look with AI-powered hairstyle recommendations. 
            Upload your photo and discover the perfect hairstyle for you.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link 
              to="/login"
              className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-8 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
            >
              Login
            </Link>
            <Link 
              to="/signup"
              className="bg-transparent hover:bg-purple-600 text-purple-600 hover:text-white font-bold py-3 px-8 rounded-lg border-2 border-purple-600 transition duration-300 ease-in-out transform hover:scale-105"
            >
              Sign Up
            </Link>
          </div>
          
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📸</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Upload Photo</h3>
              <p className="text-gray-600">Simply upload your photo to get started</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🤖</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">AI Analysis</h3>
              <p className="text-gray-600">Our AI analyzes your face shape and features</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">✨</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Get Results</h3>
              <p className="text-gray-600">Receive personalized hairstyle recommendations</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;