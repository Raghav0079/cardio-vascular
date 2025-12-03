# Supabase Database Setup Guide

This guide explains how to set up Supabase database integration for CardioPredict Pro.

## 1. Create Supabase Account and Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up for a free account
3. Create a new project
4. Wait for the project to be fully initialized

## 2. Set Up Database Schema

1. Go to the SQL Editor in your Supabase dashboard
2. Copy and paste the SQL from `database_schema.sql`
3. Run the SQL to create the required tables

## 3. Get API Credentials

1. Go to Settings → API in your Supabase dashboard
2. Copy the following values:
   - **Project URL** (something like `https://your-project-ref.supabase.co`)
   - **Anon/Public Key** (starts with `eyJ...`)

## 4. Configure Environment Variables

### For Local Development:
Update your `.env` file:
```env
WANDB_API_KEY=74ae71990b9b323813269dc56b9c2fe2f105cefa
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=eyJ...your-anon-key...
```

### For Hugging Face Spaces:
1. Go to your HF Space settings
2. Add these environment variables:
   - `SUPABASE_URL`: Your project URL
   - `SUPABASE_ANON_KEY`: Your anon key

## 5. Database Features

Once configured, your CardioPredict Pro app will:

- ✅ **Record Every Prediction**: All patient data and AI predictions saved permanently
- ✅ **Analytics Dashboard**: Track usage patterns and risk trends  
- ✅ **Historical Records**: Access previous predictions for analysis
- ✅ **Data Security**: Supabase provides enterprise-grade security
- ✅ **Real-time Updates**: Instant data synchronization

## 6. Privacy and Compliance

- Patient names are stored securely in Supabase
- All data is encrypted in transit and at rest
- Access controlled via API keys
- Compliant with healthcare data standards
- Optional anonymization for research use

## 7. Graceful Fallback

The system gracefully handles missing database credentials:
- App continues to work without database
- Warnings shown in console
- No functionality is broken
- Easy to enable later by adding credentials

## 8. Database Analytics

Access these analytics through the integrated functions:
- Total predictions count
- High-risk prediction percentage
- Average patient age
- Risk distribution patterns
- Usage trends over time

## Troubleshooting

### Connection Issues:
- Verify URL format includes `https://`
- Ensure anon key is complete
- Check Supabase project status
- Verify API limits not exceeded

### Permission Issues:
- Anon key should have table read/write access
- RLS policies may need adjustment
- Check table permissions in Supabase

### Data Issues:
- Table schema must match exactly
- Column types must be compatible
- Check for required fields