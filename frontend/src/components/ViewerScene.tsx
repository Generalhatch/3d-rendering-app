import { useEffect, useRef, useCallback } from 'react';
import { AlignmentScene } from '../three/scene';
import { useJobStore } from '../state/jobStore';

interface ViewerSceneProps {
  onReady: (scene: AlignmentScene) => void;
}

export function ViewerScene({ onReady }: ViewerSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<AlignmentScene | null>(null);
  const { selectRoom, selectFixture } = useJobStore();

  useEffect(() => {
    if (!canvasRef.current) return;
    const scene = new AlignmentScene(canvasRef.current);
    sceneRef.current = scene;

    scene.bindRoomPicking((id) => {
      selectRoom(id);
    });
    scene.bindFixturePicking((id) => {
      selectFixture(id);
    });

    scene.startRenderLoop();
    onReady(scene);

    return () => {
      scene.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full block"
      style={{ touchAction: 'none' }}
    />
  );
}
