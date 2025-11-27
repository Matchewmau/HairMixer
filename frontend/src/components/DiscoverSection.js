import React from 'react';
import HairstyleCard from './HairstyleCard';

const DiscoverSection = () => {
  const hairstyles = [
    {
      id: 1,
      image: '/dashboard/casual.png',
      title: 'Casual Everyday',
      description: 'Perfect for daily wear with a relaxed, effortless vibe that suits any casual occasion.',
      category: 'Casual',
      onClick: () => console.log('Casual style selected')
    },
    {
      id: 2,
      image: '/dashboard/trendy.png',
      title: 'Modern Trendy',
      description: 'Stay ahead of fashion with contemporary cuts that make a bold statement.',
      category: 'Trendy',
      onClick: () => console.log('Trendy style selected')
    },
    {
      id: 3,
      image: '/dashboard/formal.jpg',
      title: 'Professional Formal',
      description: 'Sophisticated looks perfect for business meetings and formal events.',
      category: 'Formal',
      onClick: () => console.log('Formal style selected')
    }
  ];

  return (
    <section className="bg-transparent">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="mb-4">
            <span className="inline-block bg-blue-500/20 text-blue-300 px-4 py-2 rounded-full text-sm font-medium border border-blue-500/30 backdrop-blur-sm">
              Featured Collection
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent">
            Discover Your Next Look
          </h2>
          <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Explore our curated collection of hairstyles designed to match every occasion and personality
          </p>
        </div>

        {/* Hairstyle Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {hairstyles.map((style) => (
            <HairstyleCard
              key={style.id}
              image={style.image}
              title={style.title}
              description={style.description}
              category={style.category}
              onClick={style.onClick}
            />
          ))}
        </div>

        {/* View All Button */}
        <div className="text-center mt-16">
          <button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-4 px-10 rounded-lg text-lg transition-all duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-2xl border border-blue-500/30">
            View All Styles
          </button>
        </div>
      </div>
    </section>
  );
};

export default DiscoverSection;