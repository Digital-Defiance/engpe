import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts', 'clinic/src/**/*.test.ts'],
    // The Metal command queue and the double-buffered channel are process-wide
    // resources; parallel workers contend for the GPU and make the gates flaky.
    fileParallelism: false,
  },
});
