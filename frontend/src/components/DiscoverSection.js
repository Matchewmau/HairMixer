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
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Discover Your Next Look
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
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
          <button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold py-4 px-10 rounded-lg transition duration-300 ease-in-out transform hover:scale-105 shadow-lg hover:shadow-purple-500/25">
            View All Styles
          </button>
        </div>
      </div>
    </section>
  );
};

export default DiscoverSection;