import interlogolos from '../../Data/Tracks/interlogolos.track.json';
import track2 from '../../Data/Tracks/track-2.track.json';
import track3 from '../../Data/Tracks/track-3.track.json';

export const bundledTracks = [
  { id: 'track-2', label: 'Track 2', data: track2 },
  { id: 'track-3', label: 'Track 3', data: track3 },
  { id: 'interlogolos', label: 'Interlogolos', data: interlogolos }
] as const;
