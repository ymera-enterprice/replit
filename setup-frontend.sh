
#!/bin/bash

echo "🚀 Setting up YMERA Frontend..."

# Create client directory if it doesn't exist
mkdir -p client/src/{components,pages,hooks,services,utils}

# Install dependencies
cd client

echo "📦 Installing dependencies..."
npm install

echo "🎨 Setting up Tailwind CSS..."
npx tailwindcss init -p

# Create tailwind config
cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ymera-gold': '#DAA520',
        'ymera-dark': '#0a0a0a',
      }
    },
  },
  plugins: [],
}
EOF

echo "✅ Frontend setup complete!"
echo "🔧 To start development server: cd client && npm run dev"
